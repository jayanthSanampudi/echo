"""Sentiment-arc endpoint — per-chapter scores for visualization."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from echomind_core.db import session_scope
from echomind_core.models import Book
from echomind_core.schemas import SentimentArcResponse, SentimentPoint

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.get("/book/{slug}", response_model=SentimentArcResponse, summary="Per-chapter sentiment")
def sentiment_arc(slug: str) -> SentimentArcResponse:
    with session_scope() as s:
        book = s.scalars(
            select(Book).options(selectinload(Book.chapters)).where(Book.slug == slug)
        ).first()
        if not book:
            raise HTTPException(status_code=404, detail=f"book {slug!r} not found")

        points: list[SentimentPoint] = []
        for ch in sorted(book.chapters, key=lambda c: c.idx):
            label = (ch.sentiment_label or "neutral").lower()
            if label not in ("negative", "neutral", "positive"):
                label = "neutral"
            points.append(
                SentimentPoint(
                    chapter_idx=ch.idx,
                    start_sec=ch.start_sec,
                    end_sec=ch.end_sec,
                    score=float(ch.sentiment_score or 0.0),
                    label=label,  # type: ignore[arg-type]
                )
            )

        return SentimentArcResponse(book_id=book.id, points=points)
