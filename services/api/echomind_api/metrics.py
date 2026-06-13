"""Prometheus metrics — exposed at /metrics."""

from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from echomind_core.config import get_settings

REQUEST_COUNT = Counter(
    "echomind_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "echomind_request_latency_seconds",
    "Request latency in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def setup_metrics(app: FastAPI) -> None:
    if not get_settings().enable_prometheus:
        return

    @app.middleware("http")
    async def _instrument(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - t0
        path = request.url.path
        # avoid metric explosion on dynamic path params — coarsen book/user IDs
        coarse = _coarsen_path(path)
        REQUEST_COUNT.labels(request.method, coarse, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.method, coarse).observe(latency)
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _coarsen_path(path: str) -> str:
    """Replace IDs / slugs in path with placeholders to keep cardinality bounded."""
    parts = path.split("/")
    out = []
    for p in parts:
        if not p:
            out.append(p)
        elif p.startswith("u_") or p.startswith("book_"):
            out.append("{id}")
        elif len(p) == 36 and p.count("-") == 4:
            out.append("{uuid}")
        else:
            out.append(p)
    return "/".join(out)
