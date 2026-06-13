"""API health-endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from echomind_api.main import create_app


def test_health_returns_ok(db, fake_vector_store):
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_returns_check_results(db, fake_vector_store):
    client = TestClient(create_app())
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    payload = r.json()
    assert "checks" in payload
    assert set(payload["checks"]) == {"db", "qdrant"}
