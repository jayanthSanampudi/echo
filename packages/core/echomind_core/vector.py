"""Qdrant vector store wrapper.

Supports both local file-based mode (no server required) and remote server mode.
The local mode is ideal for laptop development and CI runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from echomind_core.config import get_settings
from echomind_core.logging import get_logger

logger = get_logger(__name__)

TEXT_COLLECTION = "echomind_text"
AUDIO_COLLECTION = "echomind_audio"


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    """Thin wrapper around qdrant-client. One instance per process."""

    def __init__(self, client: QdrantClient | None = None) -> None:
        s = get_settings()
        if client is not None:
            self.client = client
        elif s.qdrant_mode == "local":
            logger.info("qdrant.init", mode="local", path=s.qdrant_path)
            self.client = QdrantClient(path=s.qdrant_path)
        else:
            logger.info("qdrant.init", mode="server", url=s.qdrant_url)
            self.client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)

    # ─── Collection management ───────────────────────────────────────────────

    def ensure_collection(self, name: str, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if name in existing:
            return
        logger.info("qdrant.create_collection", name=name, dim=dim)
        self.client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    def ensure_default_collections(self) -> None:
        s = get_settings()
        self.ensure_collection(TEXT_COLLECTION, s.text_embed_dim)
        self.ensure_collection(AUDIO_COLLECTION, s.audio_embed_dim)

    # ─── Writes ──────────────────────────────────────────────────────────────

    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        points = [
            qmodels.PointStruct(id=_id, vector=vec, payload=payload)
            for _id, vec, payload in zip(ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=collection, points=points)

    # ─── Reads ───────────────────────────────────────────────────────────────

    def search(
        self,
        collection: str,
        vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        flt = self._build_filter(filters) if filters else None
        result = self.client.search(
            collection_name=collection, query_vector=vector, query_filter=flt, limit=k
        )
        return [
            VectorHit(id=str(r.id), score=float(r.score), payload=dict(r.payload or {}))
            for r in result
        ]

    def delete(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return
        self.client.delete(
            collection_name=collection,
            points_selector=qmodels.PointIdsList(points=ids),  # type: ignore[arg-type]
        )

    def count(self, collection: str) -> int:
        return self.client.count(collection_name=collection, exact=True).count

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> qmodels.Filter:
        conditions = [
            qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
            for k, v in filters.items()
            if v is not None
        ]
        return qmodels.Filter(must=conditions)
