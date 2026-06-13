"""FastAPI app entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from echomind_api.deps import vector_store
from echomind_api.metrics import setup_metrics
from echomind_api.middleware import RequestLoggingMiddleware
from echomind_api.routes import (
    books,
    health,
    qa,
    recommend,
    search,
    sentiment,
    upload,
    voice_match,
)
from echomind_core.config import get_settings
from echomind_core.db import create_all
from echomind_core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.api_log_level)
    logger.info("api.startup", version="0.1.0")

    # ensure DB + vector collections exist
    create_all()
    vector_store().ensure_default_collections()
    logger.info("api.ready")
    yield
    logger.info("api.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="EchoMind API",
        description="AI-powered audiobook intelligence — search, Q&A, recommendations, voice matching.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    setup_metrics(app)

    # routes
    app.include_router(health.router)
    app.include_router(books.router, prefix="/v1")
    app.include_router(search.router, prefix="/v1")
    app.include_router(qa.router, prefix="/v1")
    app.include_router(recommend.router, prefix="/v1")
    app.include_router(voice_match.router, prefix="/v1")
    app.include_router(sentiment.router, prefix="/v1")
    app.include_router(upload.router, prefix="/v1")

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):  # noqa: ARG001
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()
