"""Database round-trip tests."""

from __future__ import annotations

from sqlalchemy import select

from echomind_core.db import session_scope
from echomind_core.models import Book, Chapter


def test_book_create_and_query(db):
    with session_scope() as s:
        b = Book(slug="test-book", title="Test Book", author="X")
        s.add(b)
        s.flush()
        bid = b.id

    with session_scope() as s:
        loaded = s.scalars(select(Book).where(Book.id == bid)).first()
        assert loaded is not None
        assert loaded.title == "Test Book"


def test_chapter_cascade_delete(db):
    with session_scope() as s:
        b = Book(slug="cascade", title="Cascade")
        b.chapters = [
            Chapter(idx=0, start_sec=0, end_sec=60, transcript="hi"),
            Chapter(idx=1, start_sec=60, end_sec=120, transcript="hello"),
        ]
        s.add(b)
        s.flush()
        bid = b.id

    with session_scope() as s:
        book = s.get(Book, bid)
        assert book is not None
        s.delete(book)

    with session_scope() as s:
        remaining = s.scalars(select(Chapter)).all()
        assert len(remaining) == 0
