"""Voice-style matching — upload an audio sample and find similar narrators."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select

from echomind_api.deps import audio_embedder, vector_store
from echomind_core.audio import DEFAULT_SR, load_audio
from echomind_core.db import session_scope
from echomind_core.models import Book
from echomind_core.schemas import VoiceMatchHit, VoiceMatchResponse
from echomind_core.vector import AUDIO_COLLECTION, VectorStore
from echomind_ml.embed_audio import AudioEmbedder

router = APIRouter(tags=["voice"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


@router.post(
    "/voice-match",
    response_model=VoiceMatchResponse,
    summary="Find books with similar narration style",
)
async def voice_match(
    sample: UploadFile = File(..., description="Audio file (wav/mp3/m4a)"),
    k: int = 10,
    embedder: AudioEmbedder = Depends(audio_embedder),
    store: VectorStore = Depends(vector_store),
) -> VoiceMatchResponse:
    raw = await sample.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")

    # save to temp file because librosa loaders work on paths
    suffix = Path(sample.filename or "sample.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        samples = load_audio(tmp_path, sr=DEFAULT_SR, mono=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    vec = embedder.embed_one(samples)
    hits = store.search(collection=AUDIO_COLLECTION, vector=vec, k=k)
    if not hits:
        return VoiceMatchResponse(hits=[])

    book_ids = [str(h.payload.get("book_id", "")) for h in hits]
    with session_scope() as sess:
        books = {b.id: b for b in sess.scalars(select(Book).where(Book.id.in_(book_ids))).all()}

    out: list[VoiceMatchHit] = []
    for h in hits:
        bid = str(h.payload.get("book_id", ""))
        b = books.get(bid)
        if b is None:
            continue
        out.append(VoiceMatchHit(book_id=bid, title=b.title, narrator=b.narrator, score=h.score))
    return VoiceMatchResponse(hits=out)
