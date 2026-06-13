"""Pydantic schema validation."""

from __future__ import annotations

import pytest

from echomind_core.schemas import (
    QARequest,
    SearchRequest,
    SearchResponse,
    SentimentPoint,
)


def test_search_request_requires_query():
    with pytest.raises(Exception):
        SearchRequest(query="", k=10)


def test_search_request_k_bounded():
    with pytest.raises(Exception):
        SearchRequest(query="hi", k=0)
    with pytest.raises(Exception):
        SearchRequest(query="hi", k=100)


def test_search_request_defaults():
    r = SearchRequest(query="frankenstein on the glacier")
    assert r.k == 10
    assert r.mode == "text"


def test_search_response_serializes():
    r = SearchResponse(query="x", hits=[], latency_ms=1.0)
    assert r.model_dump()["query"] == "x"


def test_qa_request_validates_question():
    with pytest.raises(Exception):
        QARequest(book_id="b", question="", k=5)


def test_sentiment_point_label_constrained():
    SentimentPoint(chapter_idx=0, start_sec=0.0, end_sec=10.0, score=0.4, label="positive")
    with pytest.raises(Exception):
        SentimentPoint(chapter_idx=0, start_sec=0.0, end_sec=10.0, score=0.4, label="ecstatic")  # type: ignore[arg-type]
