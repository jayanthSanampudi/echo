"""Full ingestion pipeline for one audiobook."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select

from echomind_core.audio import DEFAULT_SR, duration_sec, load_audio
from echomind_core.db import session_scope
from echomind_core.logging import get_logger
from echomind_core.models import Book, Chapter
from echomind_core.vector import AUDIO_COLLECTION, TEXT_COLLECTION, VectorStore
from echomind_ml.asr import WhisperASR
from echomind_ml.chapter_seg import (
    ChapterBoundary,
    chapter_text,
    detect_long_silences,
    segment_by_transcript,
)
from echomind_ml.embed_audio import AudioEmbedder
from echomind_ml.embed_text import TextEmbedder
from echomind_ml.sentiment import SentimentClassifier

logger = get_logger(__name__)


@dataclass
class IngestResult:
    book_id: str
    slug: str
    num_chapters: int
    duration_sec: float
    elapsed_sec: float


def ingest_file(
    audio_path: str | Path,
    slug: str,
    title: str,
    author: str | None = None,
    narrator: str | None = None,
    genre: str | None = None,
    language: str = "en",
) -> IngestResult:
    """Run the full pipeline for one audiobook."""
    t_start = time.perf_counter()
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(audio_path)

    logger.info("ingest.start", slug=slug, audio=str(path))
    asr = WhisperASR()
    text_emb = TextEmbedder()
    audio_emb = AudioEmbedder()
    sentiment = SentimentClassifier()
    store = VectorStore()
    store.ensure_default_collections()

    # 1. transcription
    t0 = time.perf_counter()
    segments = asr.transcribe(path)
    logger.info("ingest.transcribed", n_segments=len(segments), sec=time.perf_counter() - t0)

    # 2. chapter segmentation
    samples = load_audio(path, sr=DEFAULT_SR, mono=True)
    silence_breaks = detect_long_silences(samples, sr=DEFAULT_SR)
    chapters = segment_by_transcript(segments, silence_boundaries=silence_breaks)
    logger.info("ingest.chapters", n=len(chapters))

    # 3. persist book + chapters
    book_id = _persist_book(
        slug=slug,
        title=title,
        author=author,
        narrator=narrator,
        genre=genre,
        language=language,
        duration=duration_sec(path),
        audio_path=str(path),
    )

    text_payloads, text_ids, text_vectors = [], [], []
    for cb in chapters:
        ctext = chapter_text(cb, segments)
        sent = sentiment.classify(ctext[:4000])
        chapter_id = _persist_chapter(
            book_id=book_id,
            cb=cb,
            transcript=ctext,
            sentiment_score=sent.score,
            sentiment_label=sent.label,
        )
        # embed the chapter for retrieval
        if ctext.strip():
            text_vectors.append(text_emb.embed_one(ctext[:3000]))
            text_ids.append(chapter_id)
            text_payloads.append(
                {
                    "book_id": book_id,
                    "chapter_id": chapter_id,
                    "chapter_idx": cb.idx,
                    "start_sec": cb.start_sec,
                    "end_sec": cb.end_sec,
                    "title": title,
                    "author": author,
                    "language": language,
                    "genre": genre,
                    "text": ctext[:1000],
                }
            )

    if text_vectors:
        store.upsert(
            collection=TEXT_COLLECTION,
            ids=text_ids,
            vectors=text_vectors,
            payloads=text_payloads,
        )

    # 4. book-level audio embedding (first ~30s used as voice fingerprint)
    sample_window = samples[: DEFAULT_SR * 30]
    if sample_window.size > 0:
        a_vec = audio_emb.embed_one(sample_window)
        store.upsert(
            collection=AUDIO_COLLECTION,
            ids=[book_id],
            vectors=[a_vec],
            payloads=[
                {
                    "book_id": book_id,
                    "title": title,
                    "narrator": narrator,
                    "language": language,
                    "genre": genre,
                }
            ],
        )

    elapsed = time.perf_counter() - t_start
    logger.info("ingest.done", slug=slug, elapsed_sec=elapsed, chapters=len(chapters))
    return IngestResult(
        book_id=book_id,
        slug=slug,
        num_chapters=len(chapters),
        duration_sec=duration_sec(path),
        elapsed_sec=elapsed,
    )


# ─── Persistence helpers ─────────────────────────────────────────────────────


def _persist_book(
    slug: str,
    title: str,
    author: str | None,
    narrator: str | None,
    genre: str | None,
    language: str,
    duration: float,
    audio_path: str,
) -> str:
    with session_scope() as s:
        existing = s.scalars(select(Book).where(Book.slug == slug)).first()
        if existing:
            existing.title = title
            existing.author = author
            existing.narrator = narrator
            existing.genre = genre
            existing.language = language
            existing.duration_sec = duration
            existing.audio_path = audio_path
            return existing.id

        book = Book(
            id=str(uuid.uuid4()),
            slug=slug,
            title=title,
            author=author,
            narrator=narrator,
            genre=genre,
            language=language,
            duration_sec=duration,
            audio_path=audio_path,
        )
        s.add(book)
        s.flush()
        return book.id


def _persist_chapter(
    book_id: str,
    cb: ChapterBoundary,
    transcript: str,
    sentiment_score: float,
    sentiment_label: str,
) -> str:
    with session_scope() as s:
        chapter = Chapter(
            id=str(uuid.uuid4()),
            book_id=book_id,
            idx=cb.idx,
            title=cb.title,
            start_sec=cb.start_sec,
            end_sec=cb.end_sec,
            transcript=transcript,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
        )
        s.add(chapter)
        s.flush()
        return chapter.id


# ─── Ingest from raw text (no audio) ─────────────────────────────────────────

def ingest_text(
    slug: str,
    title: str,
    chapters_text: list[str],
    author: str | None = None,
    genre: str | None = None,
    language: str = "en",
) -> IngestResult:
    """Index a book whose source is already text (e.g., Project Gutenberg).
    Useful for seeding the catalog without 100MB of audio downloads.
    """
    t_start = time.perf_counter()
    text_emb = TextEmbedder()
    sentiment = SentimentClassifier()
    store = VectorStore()
    store.ensure_default_collections()

    book_id = _persist_book(
        slug=slug,
        title=title,
        author=author,
        narrator=None,
        genre=genre,
        language=language,
        duration=0.0,
        audio_path=None or "",
    )

    text_ids, text_vectors, payloads = [], [], []
    for i, ctext in enumerate(chapters_text):
        sent = sentiment.classify(ctext[:4000])
        cb = ChapterBoundary(
            idx=i, start_sec=float(i * 600), end_sec=float((i + 1) * 600), title=None
        )
        chapter_id = _persist_chapter(
            book_id=book_id,
            cb=cb,
            transcript=ctext,
            sentiment_score=sent.score,
            sentiment_label=sent.label,
        )
        if ctext.strip():
            text_vectors.append(text_emb.embed_one(ctext[:3000]))
            text_ids.append(chapter_id)
            payloads.append(
                {
                    "book_id": book_id,
                    "chapter_id": chapter_id,
                    "chapter_idx": i,
                    "start_sec": cb.start_sec,
                    "end_sec": cb.end_sec,
                    "title": title,
                    "author": author,
                    "genre": genre,
                    "language": language,
                    "text": ctext[:1000],
                }
            )

    if text_vectors:
        store.upsert(TEXT_COLLECTION, text_ids, text_vectors, payloads)

    elapsed = time.perf_counter() - t_start
    logger.info("ingest_text.done", slug=slug, chapters=len(chapters_text), elapsed_sec=elapsed)
    return IngestResult(
        book_id=book_id,
        slug=slug,
        num_chapters=len(chapters_text),
        duration_sec=0.0,
        elapsed_sec=elapsed,
    )


# ─── Re-embedding utility ────────────────────────────────────────────────────

def reembed_all() -> int:
    """Re-embed every chapter in the DB. Run after changing the embedding model."""
    text_emb = TextEmbedder()
    store = VectorStore()
    store.ensure_default_collections()
    n = 0

    with session_scope() as s:
        chapters = s.scalars(select(Chapter)).all()
        # snapshot data before scope closes — vectors are independent of the session
        rows: list[tuple[str, str, int, float, float, str]] = [
            (c.id, c.book_id, c.idx, c.start_sec, c.end_sec, c.transcript or "")
            for c in chapters
        ]

    if not rows:
        return 0

    # batch in groups of 64
    BATCH = 64
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        texts = [t for *_, t in batch]
        vectors = [text_emb.embed_one(t[:3000]) if t.strip() else [0.0] * text_emb.dim for t in texts]
        ids = [r[0] for r in batch]
        payloads = [
            {
                "chapter_id": cid,
                "book_id": bid,
                "chapter_idx": idx,
                "start_sec": s_sec,
                "end_sec": e_sec,
                "text": txt[:1000],
            }
            for cid, bid, idx, s_sec, e_sec, txt in batch
        ]
        store.upsert(TEXT_COLLECTION, ids, vectors, payloads)
        n += len(batch)
    return n


def _np_zeros(d: int) -> np.ndarray:
    return np.zeros(d, dtype=np.float32)
