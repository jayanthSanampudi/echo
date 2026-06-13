"""Book catalog endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from echomind_api.main import create_app
from echomind_core.db import session_scope
from echomind_core.models import Book, Chapter


def test_list_and_get_book(db, fake_vector_store):
    with session_scope() as s:
        b = Book(slug="alice", title="Alice in Wonderland", author="Lewis Carroll", genre="fantasy")
        b.chapters = [
            Chapter(idx=0, start_sec=0, end_sec=60, transcript="Down the rabbit hole."),
            Chapter(idx=1, start_sec=60, end_sec=120, transcript="A mad tea party."),
        ]
        s.add(b)

    client = TestClient(create_app())

    r = client.get("/v1/books")
    assert r.status_code == 200
    books = r.json()
    assert any(b["slug"] == "alice" for b in books)

    r = client.get("/v1/books/alice")
    assert r.status_code == 200
    data = r.json()
    assert data["title"] == "Alice in Wonderland"
    assert len(data["chapters"]) == 2

    r = client.get("/v1/books/nope")
    assert r.status_code == 404
