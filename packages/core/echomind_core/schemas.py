"""Pydantic schemas — used at the API boundary and between services."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ─── Books / chapters ────────────────────────────────────────────────────────

class ChapterOut(_Base):
    id: str
    idx: int
    title: str | None = None
    start_sec: float
    end_sec: float
    sentiment_score: float | None = None
    sentiment_label: str | None = None


class BookOut(_Base):
    id: str
    slug: str
    title: str
    author: str | None = None
    narrator: str | None = None
    description: str | None = None
    genre: str | None = None
    language: str = "en"
    duration_sec: float = 0.0
    chapters: list[ChapterOut] = Field(default_factory=list)
    created_at: datetime | None = None


# ─── Search ──────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    k: int = Field(default=10, ge=1, le=50)
    genre: str | None = None
    language: str | None = None
    mode: Literal["text", "audio", "hybrid"] = "text"


class SearchHit(BaseModel):
    book_id: str
    chapter_id: str | None = None
    score: float
    snippet: str
    title: str
    author: str | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    latency_ms: float


# ─── Q&A (RAG) ───────────────────────────────────────────────────────────────

class QARequest(BaseModel):
    book_id: str
    question: str = Field(..., min_length=1, max_length=2048)
    k: int = Field(default=5, ge=1, le=20)


class QACitation(BaseModel):
    chapter_idx: int
    start_sec: float
    end_sec: float
    text: str


class QAResponse(BaseModel):
    book_id: str
    question: str
    answer: str
    citations: list[QACitation]
    latency_ms: float


# ─── Recommendations ─────────────────────────────────────────────────────────

class RecommendResponse(BaseModel):
    user_id: str
    books: list[BookOut]
    explain: list[str] = Field(default_factory=list)


# ─── Voice-style matching ────────────────────────────────────────────────────

class VoiceMatchHit(BaseModel):
    book_id: str
    title: str
    narrator: str | None = None
    score: float


class VoiceMatchResponse(BaseModel):
    hits: list[VoiceMatchHit]


# ─── Sentiment arc ───────────────────────────────────────────────────────────

class SentimentPoint(BaseModel):
    chapter_idx: int
    start_sec: float
    end_sec: float
    score: float  # -1 negative .. +1 positive
    label: Literal["negative", "neutral", "positive"]


class SentimentArcResponse(BaseModel):
    book_id: str
    points: list[SentimentPoint]


# ─── Upload / ingest ─────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    slug: str
    title: str
    author: str | None = None
    narrator: str | None = None
    genre: str | None = None
    language: str = "en"
    audio_url: str | None = None


class IngestResponse(BaseModel):
    book_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    message: str | None = None
