"""Audiobook ingestion trigger.

Production: enqueue a job to the worker (via Redis or SQS).
MVP: run inline if the audio is small; otherwise return 202 with a job ID.
"""

from __future__ import annotations

from fastapi import APIRouter

from echomind_core.db import session_scope
from echomind_core.logging import get_logger
from echomind_core.models import Book
from echomind_core.schemas import IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])
logger = get_logger(__name__)


@router.post("/ingest", response_model=IngestResponse, summary="Trigger ingestion of an audiobook")
def ingest(req: IngestRequest) -> IngestResponse:
    """Create a Book row in 'queued' state. The worker watches and processes."""
    with session_scope() as s:
        existing = s.query(Book).filter(Book.slug == req.slug).first()
        if existing:
            return IngestResponse(
                book_id=existing.id, status="queued", message="book already exists"
            )
        book = Book(
            slug=req.slug,
            title=req.title,
            author=req.author,
            narrator=req.narrator,
            genre=req.genre,
            language=req.language,
            meta={"audio_url": req.audio_url} if req.audio_url else {},
        )
        s.add(book)
        s.flush()
        logger.info("ingest.queued", book_id=book.id, slug=req.slug)
        return IngestResponse(book_id=book.id, status="queued")
