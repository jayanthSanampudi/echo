"""Text embedding via sentence-transformers."""

from __future__ import annotations

from typing import Any

import numpy as np

from echomind_core.config import get_settings
from echomind_core.logging import get_logger

logger = get_logger(__name__)


class TextEmbedder:
    """Wraps a sentence-transformers model. 384-dim by default (MiniLM-L6-v2)."""

    _model: Any = None
    _model_name: str | None = None

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().text_embed_model

    def _load(self) -> Any:
        cls = type(self)
        if cls._model is None or cls._model_name != self.model_name:
            from sentence_transformers import SentenceTransformer

            cache_dir = str(get_settings().model_cache_path)
            logger.info("text_embed.load", model=self.model_name, cache=cache_dir)
            cls._model = SentenceTransformer(self.model_name, cache_folder=cache_dir)
            cls._model_name = self.model_name
        return cls._model

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def embed(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Embed one or many texts. Returns (N, dim) float32 array, L2-normalized."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        model = self._load()
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text and return as a plain Python list (Qdrant-friendly)."""
        return self.embed(text)[0].tolist()
