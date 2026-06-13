"""FastAPI dependency factories — singletons for embedders and pipelines."""

from __future__ import annotations

from functools import lru_cache

from echomind_core.vector import VectorStore
from echomind_ml.embed_audio import AudioEmbedder
from echomind_ml.embed_text import TextEmbedder
from echomind_ml.rag import RAGPipeline
from echomind_ml.sentiment import SentimentClassifier


@lru_cache(maxsize=1)
def text_embedder() -> TextEmbedder:
    return TextEmbedder()


@lru_cache(maxsize=1)
def audio_embedder() -> AudioEmbedder:
    return AudioEmbedder()


@lru_cache(maxsize=1)
def vector_store() -> VectorStore:
    return VectorStore()


@lru_cache(maxsize=1)
def rag_pipeline() -> RAGPipeline:
    return RAGPipeline(embedder=text_embedder(), store=vector_store())


@lru_cache(maxsize=1)
def sentiment_classifier() -> SentimentClassifier:
    return SentimentClassifier()
