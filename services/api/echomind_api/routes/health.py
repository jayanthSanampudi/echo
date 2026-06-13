"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from echomind_core.db import get_engine
from echomind_core.vector import VectorStore

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness — checks DB + vector store")
def ready() -> JSONResponse:
    checks = {"db": False, "qdrant": False}
    try:
        with get_engine().connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        checks["db"] = True
    except Exception:
        pass
    try:
        VectorStore().client.get_collections()
        checks["qdrant"] = True
    except Exception:
        pass

    ok = all(checks.values())
    return JSONResponse(
        status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"ready": ok, "checks": checks},
    )
