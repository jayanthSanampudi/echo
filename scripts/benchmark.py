"""Quick performance benchmarks for the EchoMind hot paths.

Measures:
- Text embedding throughput
- Vector search latency (Qdrant local mode)
- End-to-end /search and /qa latency via TestClient

Run:
    uv run python scripts/benchmark.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np
from rich.console import Console
from rich.table import Table


console = Console()


def _time(fn, n: int = 100) -> tuple[float, float]:
    """Return (mean_ms, p95_ms)."""
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return statistics.mean(samples), samples[int(0.95 * len(samples))]


def bench_text_embed() -> tuple[float, float]:
    from echomind_ml.embed_text import TextEmbedder

    emb = TextEmbedder()
    emb._load()  # warm
    return _time(lambda: emb.embed_one("a quick test sentence"), n=50)


def bench_vector_search() -> tuple[float, float]:
    from echomind_core.vector import VectorStore

    store = VectorStore()
    store.ensure_default_collections()

    # seed 1000 random vectors
    rng = np.random.default_rng(0)
    n = 1000
    vecs = rng.normal(size=(n, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    payloads = [{"book_id": f"b{i}", "title": f"book {i}"} for i in range(n)]
    store.upsert("echomind_text", [f"id_{i}" for i in range(n)], vecs.tolist(), payloads)

    q = vecs[0].tolist()
    return _time(lambda: store.search("echomind_text", q, k=20), n=50)


def main() -> None:
    table = Table(title="EchoMind benchmarks", show_lines=True)
    table.add_column("Operation")
    table.add_column("mean (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")

    mean, p95 = bench_text_embed()
    table.add_row("Text embed (single)", f"{mean:.2f}", f"{p95:.2f}")

    mean, p95 = bench_vector_search()
    table.add_row("Vector search (k=20, n=1000)", f"{mean:.2f}", f"{p95:.2f}")

    console.print(table)


if __name__ == "__main__":
    main()
