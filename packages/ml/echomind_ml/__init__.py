"""EchoMind ML wrappers: ASR, embeddings, sentiment, segmentation, RAG, recommender."""

from echomind_ml.asr import WhisperASR, TranscriptSegment
from echomind_ml.embed_text import TextEmbedder
from echomind_ml.embed_audio import AudioEmbedder
from echomind_ml.sentiment import SentimentClassifier, SentimentResult

__version__ = "0.1.0"
__all__ = [
    "WhisperASR",
    "TranscriptSegment",
    "TextEmbedder",
    "AudioEmbedder",
    "SentimentClassifier",
    "SentimentResult",
]
