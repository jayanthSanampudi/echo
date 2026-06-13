"""Sentiment classifier. Backed by a HuggingFace transformer with a graceful fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from echomind_core.config import get_settings
from echomind_core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SentimentResult:
    label: str  # "negative" | "neutral" | "positive"
    score: float  # -1 .. +1
    confidence: float  # raw probability of predicted class


class SentimentClassifier:
    _pipe: Any = None
    _fallback: bool = False

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().sentiment_model

    def _load(self) -> Any:
        cls = type(self)
        if cls._pipe is None and not cls._fallback:
            try:
                from transformers import pipeline

                cache_dir = str(get_settings().model_cache_path)
                logger.info("sentiment.load", model=self.model_name)
                cls._pipe = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    top_k=None,
                    model_kwargs={"cache_dir": cache_dir},
                )
            except Exception as e:
                logger.warning("sentiment.fallback", reason=str(e))
                cls._fallback = True
        return cls._pipe

    def classify(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult("neutral", 0.0, 1.0)
        pipe = self._load()
        if pipe is None:
            return self._lexicon_fallback(text)

        outputs = pipe(text[:4000])  # truncate; tokenizer also truncates internally
        # outputs is a list with one element (per input) containing list of dicts
        scores = outputs[0] if isinstance(outputs[0], list) else outputs
        return self._aggregate(scores)

    def classify_many(self, texts: list[str]) -> list[SentimentResult]:
        return [self.classify(t) for t in texts]

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(scores: list[dict[str, Any]]) -> SentimentResult:
        norm = {s["label"].lower(): float(s["score"]) for s in scores}
        # cardiffnlp uses LABEL_0/1/2 in some checkpoints — translate
        translation = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
        translated = {translation.get(k, k): v for k, v in norm.items()}
        pos = translated.get("positive", 0.0)
        neg = translated.get("negative", 0.0)
        score = pos - neg  # in [-1, 1]
        if score > 0.15:
            label = "positive"
        elif score < -0.15:
            label = "negative"
        else:
            label = "neutral"
        confidence = max(translated.values()) if translated else 0.0
        return SentimentResult(label=label, score=score, confidence=confidence)

    @staticmethod
    def _lexicon_fallback(text: str) -> SentimentResult:
        """Very small lexicon — only used when the transformer can't load."""
        positive = {"love", "great", "wonderful", "happy", "beautiful", "joy", "hope", "kind"}
        negative = {"hate", "terrible", "sad", "angry", "fear", "dark", "death", "pain"}
        tokens = {t.lower().strip(".,!?;:") for t in text.split()}
        p = len(tokens & positive)
        n = len(tokens & negative)
        total = max(p + n, 1)
        score = (p - n) / total
        label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
        return SentimentResult(label=label, score=score, confidence=0.5)
