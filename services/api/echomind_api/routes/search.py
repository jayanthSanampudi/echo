"""Semantic search across the audiobook catalog."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from echomind_api.deps import text_embedder, vector_store
from echomind_core.cache import get_cache
from echomind_core.schemas import SearchHit, SearchRequest, SearchResponse
from echomind_core.vector import TEXT_COLLECTION, VectorStore
from echomind_ml.embed_text import TextEmbedder

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse, summary="Semantic search")
def search(
    req: SearchRequest,
    embedder: TextEmbedder = Depends(text_embedder),
    store: VectorStore = Depends(vector_store),
) -> SearchResponse:
    t0 = time.perf_counter()
    cache = get_cache()
    cache_key = f"search:{req.mode}:{req.k}:{req.genre}:{req.language}:{req.query}"
    if (cached := cache.get(cache_key)) is not None:
        return SearchResponse(**cached)

    qvec = embedder.embed_one(req.query)

    filters: dict[str, object] = {}
    if req.genre:
        filters["genre"] = req.genre
    if req.language:
        filters["language"] = req.language

    hits = store.search(
        collection=TEXT_COLLECTION,
        vector=qvec,
        k=req.k,
        filters=filters or None,
    )

    response = SearchResponse(
        query=req.query,
        hits=[
            SearchHit(
                book_id=str(h.payload.get("book_id", "")),
                chapter_id=str(h.payload.get("chapter_id", "")) or None,
                score=h.score,
                snippet=str(h.payload.get("text", ""))[:300],
                title=str(h.payload.get("title", "")),
                author=str(h.payload.get("author", "")) or None,
            )
            for h in hits
        ],
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    cache.set(cache_key, response.model_dump(), ttl=120)
    return response
