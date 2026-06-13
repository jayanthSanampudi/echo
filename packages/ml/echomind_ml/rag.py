"""Chapter Q&A via retrieval-augmented generation.

The pipeline:
    1. Embed the question (TextEmbedder)
    2. Retrieve top-k chunks for the requested book from Qdrant
    3. Build a structured prompt with citations
    4. Call the LLM (real or mock) for the answer
    5. Return answer + citations
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from echomind_core.logging import get_logger
from echomind_core.vector import TEXT_COLLECTION, VectorStore
from echomind_ml.embed_text import TextEmbedder
from echomind_ml.llm import LLMClient, get_llm

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are EchoMind, an assistant that answers questions about audiobooks using only the provided context.

Rules:
- Answer concisely (1-3 sentences) unless asked for more detail.
- Quote specific phrases from the context when relevant.
- If the answer is not in the context, say "I don't see that in this book."
- Do not invent facts."""


@dataclass(frozen=True)
class RetrievedChunk:
    chapter_idx: int
    start_sec: float
    end_sec: float
    text: str
    score: float


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    chunks: list[RetrievedChunk]
    latency_ms: float


class RAGPipeline:
    def __init__(
        self,
        embedder: TextEmbedder | None = None,
        store: VectorStore | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.embedder = embedder or TextEmbedder()
        self.store = store or VectorStore()
        self.llm = llm or get_llm()

    def answer(self, book_id: str, question: str, k: int = 5) -> RAGAnswer:
        t0 = time.perf_counter()
        qvec = self.embedder.embed_one(question)
        hits = self.store.search(
            collection=TEXT_COLLECTION,
            vector=qvec,
            k=k,
            filters={"book_id": book_id},
        )
        chunks = [
            RetrievedChunk(
                chapter_idx=int(h.payload.get("chapter_idx", -1)),
                start_sec=float(h.payload.get("start_sec", 0.0)),
                end_sec=float(h.payload.get("end_sec", 0.0)),
                text=str(h.payload.get("text", "")),
                score=h.score,
            )
            for h in hits
        ]
        prompt = self._build_prompt(question, chunks)
        answer_text = self.llm.complete(SYSTEM_PROMPT, prompt, max_tokens=400)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("rag.answer", book_id=book_id, k=len(chunks), latency_ms=latency_ms)
        return RAGAnswer(answer=answer_text.strip(), chunks=chunks, latency_ms=latency_ms)

    @staticmethod
    def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
        ctx_lines = [
            f"[Chapter {c.chapter_idx} @ {c.start_sec:.0f}s] {c.text}" for c in chunks
        ]
        ctx = "\n\n".join(ctx_lines) if ctx_lines else "(no relevant context found)"
        return f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
