"""Thin HTTPX client for the EchoMind API used by every Streamlit page."""

from __future__ import annotations

import os
from typing import Any

import httpx

API_URL = os.getenv("ECHOMIND_API_URL", "http://localhost:8000")


class APIClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = (base_url or API_URL).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ─── Health / catalog ────────────────────────────────────────────────────

    def health(self) -> bool:
        try:
            r = self._client.get("/health")
            return r.status_code == 200
        except Exception:
            return False

    def list_books(self, limit: int = 50, genre: str | None = None) -> list[dict[str, Any]]:
        r = self._client.get("/v1/books", params={"limit": limit, "genre": genre})
        r.raise_for_status()
        return r.json()

    def get_book(self, slug: str) -> dict[str, Any]:
        r = self._client.get(f"/v1/books/{slug}")
        r.raise_for_status()
        return r.json()

    # ─── Features ────────────────────────────────────────────────────────────

    def search(
        self, query: str, k: int = 10, mode: str = "text", genre: str | None = None
    ) -> dict[str, Any]:
        r = self._client.post(
            "/v1/search",
            json={"query": query, "k": k, "mode": mode, "genre": genre},
        )
        r.raise_for_status()
        return r.json()

    def qa(self, book_id: str, question: str, k: int = 5) -> dict[str, Any]:
        r = self._client.post(
            "/v1/qa",
            json={"book_id": book_id, "question": question, "k": k},
        )
        r.raise_for_status()
        return r.json()

    def recommend_user(self, handle: str, k: int = 10) -> dict[str, Any]:
        r = self._client.get(f"/v1/recommend/user/{handle}", params={"k": k})
        r.raise_for_status()
        return r.json()

    def recommend_book(self, book_id: str, k: int = 10) -> dict[str, Any]:
        r = self._client.get(f"/v1/recommend/book/{book_id}", params={"k": k})
        r.raise_for_status()
        return r.json()

    def voice_match(self, audio_bytes: bytes, filename: str, k: int = 10) -> dict[str, Any]:
        files = {"sample": (filename, audio_bytes, "audio/wav")}
        r = self._client.post("/v1/voice-match", files=files, params={"k": k})
        r.raise_for_status()
        return r.json()

    def sentiment_arc(self, slug: str) -> dict[str, Any]:
        r = self._client.get(f"/v1/sentiment/book/{slug}")
        r.raise_for_status()
        return r.json()

    def ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post("/v1/ingest", json=payload)
        r.raise_for_status()
        return r.json()


def get_client() -> APIClient:
    return APIClient()
