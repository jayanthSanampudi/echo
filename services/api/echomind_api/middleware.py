"""Request logging middleware with structured logs and request IDs."""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from echomind_core.logging import get_logger

logger = get_logger("echomind_api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                "request.failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                latency_ms=latency_ms,
            )
            raise
        latency_ms = (time.perf_counter() - t0) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{latency_ms:.1f}"
        logger.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
        )
        return response
