"""RAG Q&A endpoint with mock LLM and fake vector store."""

from __future__ import annotations

from fastapi.testclient import TestClient

from echomind_api import deps
from echomind_api.main import create_app
from echomind_core.vector import TEXT_COLLECTION
from echomind_ml.llm import _MockLLM
from echomind_ml.rag import RAGPipeline


def test_qa_returns_answer_and_citations(db, fake_vector_store, fake_text_embedder, monkeypatch):
    deps.rag_pipeline.cache_clear()
    deps.text_embedder.cache_clear()
    deps.vector_store.cache_clear()

    pipeline = RAGPipeline(embedder=fake_text_embedder, store=fake_vector_store, llm=_MockLLM())
    monkeypatch.setattr(deps, "rag_pipeline", lambda: pipeline)

    # seed two chunks for book_id "frank"
    payloads = [
        {"book_id": "frank", "chapter_idx": 0, "start_sec": 0, "end_sec": 60, "text": "Victor met the creature on the glacier. It was a cold morning."},
        {"book_id": "frank", "chapter_idx": 1, "start_sec": 60, "end_sec": 120, "text": "Elizabeth wrote a letter expressing her concern."},
    ]
    vectors = [fake_text_embedder.embed_one(p["text"]) for p in payloads]
    fake_vector_store.upsert(TEXT_COLLECTION, ["c1", "c2"], vectors, payloads)

    client = TestClient(create_app())
    r = client.post("/v1/qa", json={"book_id": "frank", "question": "Where did Victor meet the creature?", "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["book_id"] == "frank"
    assert data["answer"]
    assert len(data["citations"]) >= 1
