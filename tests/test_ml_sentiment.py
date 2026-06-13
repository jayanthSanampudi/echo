"""SentimentClassifier — uses the lexicon fallback so no model download is needed."""

from __future__ import annotations

from echomind_ml.sentiment import SentimentClassifier


def _force_fallback() -> SentimentClassifier:
    sc = SentimentClassifier()
    SentimentClassifier._fallback = True
    SentimentClassifier._pipe = None
    return sc


def test_positive_text_scores_positive():
    sc = _force_fallback()
    r = sc.classify("This was a wonderful, joyful, beautiful chapter.")
    assert r.label == "positive"
    assert r.score > 0


def test_negative_text_scores_negative():
    sc = _force_fallback()
    r = sc.classify("Pain, death, fear and darkness everywhere.")
    assert r.label == "negative"
    assert r.score < 0


def test_empty_text_returns_neutral():
    sc = _force_fallback()
    r = sc.classify("")
    assert r.label == "neutral"
    assert r.score == 0.0
