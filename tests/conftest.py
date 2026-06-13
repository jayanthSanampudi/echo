"""Shared pytest fixtures: in-memory DB, mock vector store, fake embedders."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

# Force test config BEFORE any echomind import touches settings
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["CACHE_BACKEND"] = "memory"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["QDRANT_MODE"] = "local"
os.environ["QDRANT_PATH"] = "/tmp/echomind_test_qdrant"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from echomind_core.db import create_all, drop_all, get_engine  # noqa: E402
from echomind_core.vector import VectorHit  # noqa: E402


# ─── Database ────────────────────────────────────────────────────────────────


@pytest.fixture()
def db() -> Iterator[None]:
    """Fresh in-memory SQLite per test."""
    # in-memory sqlite needs special handling with multiple connections,
    # so force a single shared connection
    create_all()
    try:
        yield
    finally:
        drop_all()
        get_engine().dispose()


# ─── Fake vector store ───────────────────────────────────────────────────────


@dataclass
class FakeVectorStore:
    """In-memory vector store with cosine similarity."""

    collections: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def ensure_collection(self, name: str, dim: int) -> None:  # noqa: ARG002
        self.collections.setdefault(name, {})

    def ensure_default_collections(self) -> None:
        self.ensure_collection("echomind_text", 384)
        self.ensure_collection("echomind_audio", 512)

    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        bucket = self.collections.setdefault(collection, {})
        for _id, vec, payload in zip(ids, vectors, payloads, strict=True):
            bucket[_id] = {"vector": np.asarray(vec, dtype=np.float32), "payload": payload}

    def search(
        self,
        collection: str,
        vector: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        bucket = self.collections.get(collection, {})
        q = np.asarray(vector, dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-12)
        results = []
        for _id, item in bucket.items():
            if filters:
                ok = all(item["payload"].get(k) == v for k, v in filters.items() if v is not None)
                if not ok:
                    continue
            v = item["vector"]
            vn = v / (np.linalg.norm(v) + 1e-12)
            score = float(np.dot(qn, vn))
            results.append(VectorHit(id=_id, score=score, payload=dict(item["payload"])))
        results.sort(key=lambda h: -h.score)
        return results[:k]

    def delete(self, collection: str, ids: list[str]) -> None:
        bucket = self.collections.get(collection, {})
        for _id in ids:
            bucket.pop(_id, None)

    def count(self, collection: str) -> int:
        return len(self.collections.get(collection, {}))


@pytest.fixture()
def fake_vector_store(monkeypatch: pytest.MonkeyPatch) -> FakeVectorStore:
    store = FakeVectorStore()
    store.ensure_default_collections()
    from echomind_core import vector as vector_module

    monkeypatch.setattr(vector_module, "VectorStore", lambda *_a, **_kw: store)
    return store


# ─── Fake text embedder ──────────────────────────────────────────────────────


class FakeTextEmbedder:
    """Deterministic embedder — bag-of-letter-hash. Tiny, fast, no model download."""

    dim: int = 32

    def embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            idx = hash(tok) % self.dim
            vec[idx] += 1.0
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec.tolist()

    def embed(self, texts: list[str] | str, batch_size: int = 32) -> np.ndarray:  # noqa: ARG002
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([np.asarray(self.embed_one(t), dtype=np.float32) for t in texts])


@pytest.fixture()
def fake_text_embedder() -> FakeTextEmbedder:
    return FakeTextEmbedder()


# ─── Module path injection ───────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _inject_paths() -> None:
    """Make sure all workspace packages are importable even without uv sync."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for sub in ("packages/core", "packages/ml", "services/api", "services/worker", "services/ui"):
        p = os.path.join(root, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
