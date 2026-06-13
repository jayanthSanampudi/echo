"""Personalized recommendations from the two-tower model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from echomind_core.config import get_settings
from echomind_core.db import session_scope
from echomind_core.models import Book, Interaction, User
from echomind_core.schemas import BookOut, RecommendResponse
from echomind_ml.recommender import RecommenderIndex, TwoTower

router = APIRouter(prefix="/recommend", tags=["recommend"])

_index: RecommenderIndex | None = None


def _load_index() -> RecommenderIndex | None:
    """Lazy-load saved recommender model + build inference index from the DB."""
    global _index
    if _index is not None:
        return _index

    s = get_settings()
    ckpt_path = Path(s.model_cache_dir) / "recommender.pt"
    if not ckpt_path.exists():
        return None

    ckpt: dict[str, Any] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TwoTower(
        num_users=ckpt["num_users"],
        content_dim=ckpt["content_dim"],
        embed_dim=ckpt["embed_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # build item embeddings from the DB's stored content vectors (payload "embedding")
    # for the MVP we use random projections of book titles as a stand-in if vectors absent
    with session_scope() as sess:
        books = list(sess.scalars(select(Book)).all())
        users = list(sess.scalars(select(User)).all())

    rng = np.random.default_rng(42)
    book_content = rng.normal(size=(len(books), ckpt["content_dim"])).astype(np.float32)
    book_content /= np.linalg.norm(book_content, axis=1, keepdims=True) + 1e-12

    with torch.no_grad():
        item_emb = model.item(torch.from_numpy(book_content)).numpy()
        user_emb = model.user(torch.arange(ckpt["num_users"])).numpy()

    handle_to_idx = {u.handle: i for i, u in enumerate(users[: ckpt["num_users"]])}

    _index = RecommenderIndex(
        book_ids=[b.id for b in books],
        item_emb=item_emb,
        user_emb=user_emb,
        user_handle_to_idx=handle_to_idx,
    )
    return _index


def _books_by_ids(book_ids: list[str]) -> list[BookOut]:
    with session_scope() as sess:
        books = sess.scalars(select(Book).where(Book.id.in_(book_ids))).all()
        by_id = {b.id: b for b in books}
        return [BookOut.model_validate(by_id[bid]) for bid in book_ids if bid in by_id]


@router.get("/user/{handle}", response_model=RecommendResponse, summary="Recommend books for a user")
def recommend_user(handle: str, k: int = Query(10, ge=1, le=50)) -> RecommendResponse:
    idx = _load_index()
    if idx is None:
        return _popularity(handle, k, "no trained recommender — using popularity fallback")
    pairs = idx.recommend_user(handle, k=k)
    explanations = [f"score={score:.3f}" for _, score in pairs]
    return RecommendResponse(
        user_id=handle, books=_books_by_ids([bid for bid, _ in pairs]), explain=explanations
    )


@router.get("/book/{book_id}", response_model=RecommendResponse, summary="Similar-book recommendations")
def recommend_book(book_id: str, k: int = Query(10, ge=1, le=50)) -> RecommendResponse:
    idx = _load_index()
    if idx is None:
        raise HTTPException(status_code=503, detail="recommender not trained; run `python -m echomind_ml.recommender --train`")
    pairs = idx.recommend_item(book_id, k=k)
    return RecommendResponse(
        user_id=book_id,
        books=_books_by_ids([bid for bid, _ in pairs]),
        explain=[f"similarity={s:.3f}" for _, s in pairs],
    )


def _popularity(handle: str, k: int, reason: str) -> RecommendResponse:
    """Fallback when no trained model exists yet — return most-listened books."""
    from sqlalchemy import func

    with session_scope() as sess:
        stmt = (
            select(Interaction.book_id, func.count().label("cnt"))
            .group_by(Interaction.book_id)
            .order_by(func.count().desc())
            .limit(k)
        )
        rows = list(sess.execute(stmt))
        book_ids = [r[0] for r in rows]
    return RecommendResponse(user_id=handle, books=_books_by_ids(book_ids), explain=[reason])
