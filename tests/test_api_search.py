"""Search endpoint — patched embedder + fake store."""

from __future__ import annotations

from fastapi.testclient import TestClient

from echomind_api import deps
from echomind_api.main import create_app
from echomind_core.vector import TEXT_COLLECTION


def test_search_returns_hits(db, fake_vector_store, fake_text_embedder, monkeypatch):
    # patch the singleton getters
    deps.text_embedder.cache_clear()
    deps.vector_store.cache_clear()
    monkeypatch.setattr(deps, "text_embedder", lambda: fake_text_embedder)
    monkeypatch.setattr(deps, "vector_store", lambda: fake_vector_store)

    # seed the fake store with three chapters
    payloads = [
        {"book_id": "b1", "title": "Frankenstein", "author": "Mary Shelley", "text": "Victor met the creature on the glacier."},
        {"book_id": "b2", "title": "Dracula", "author": "Bram Stoker", "text": "Strange shadows over Transylvania."},
        {"book_id": "b3", "title": "Pride and Prejudice", "author": "Jane Austen", "text": "Elizabeth and Mr Darcy argue politely."},
    ]
    vectors = [fake_text_embedder.embed_one(p["text"]) for p in payloads]
    fake_vector_store.upsert(TEXT_COLLECTION, ["c1", "c2", "c3"], vectors, payloads)

    client = TestClient(create_app())
    r = client.post("/v1/search", json={"query": "Victor on the glacier", "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "Victor on the glacier"
    assert len(data["hits"]) > 0
    assert data["hits"][0]["title"] == "Frankenstein"
