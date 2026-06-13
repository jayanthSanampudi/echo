"""Chapter-aware Q&A using RAG."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from echomind_api.deps import rag_pipeline
from echomind_core.schemas import QACitation, QARequest, QAResponse
from echomind_ml.rag import RAGPipeline

router = APIRouter(tags=["qa"])


@router.post("/qa", response_model=QAResponse, summary="Ask a question about a book")
def qa(req: QARequest, pipeline: RAGPipeline = Depends(rag_pipeline)) -> QAResponse:
    result = pipeline.answer(book_id=req.book_id, question=req.question, k=req.k)
    return QAResponse(
        book_id=req.book_id,
        question=req.question,
        answer=result.answer,
        citations=[
            QACitation(
                chapter_idx=c.chapter_idx,
                start_sec=c.start_sec,
                end_sec=c.end_sec,
                text=c.text[:400],
            )
            for c in result.chunks
        ],
        latency_ms=result.latency_ms,
    )
