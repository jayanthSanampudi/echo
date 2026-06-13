"""Book catalog endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from echomind_core.db import session_scope
from echomind_core.models import Book
from echomind_core.schemas import BookOut

router = APIRouter(prefix="/books", tags=["books"])


@router.get("", response_model=list[BookOut], summary="List books")
def list_books(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    genre: str | None = None,
) -> list[BookOut]:
    with session_scope() as s:
        stmt = select(Book).options(selectinload(Book.chapters)).order_by(Book.title)
        if genre:
            stmt = stmt.where(Book.genre == genre)
        stmt = stmt.limit(limit).offset(offset)
        return [BookOut.model_validate(b) for b in s.scalars(stmt).all()]


@router.get("/{slug}", response_model=BookOut, summary="Book detail")
def get_book(slug: str) -> BookOut:
    with session_scope() as s:
        stmt = select(Book).options(selectinload(Book.chapters)).where(Book.slug == slug)
        book = s.scalars(stmt).first()
        if not book:
            raise HTTPException(status_code=404, detail=f"book {slug!r} not found")
        return BookOut.model_validate(book)
