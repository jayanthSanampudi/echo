# Design notes

A written record of the non-obvious trade-offs.

## 1. Why a single repo, not five

EchoMind has five deployables (api, worker, ui, qdrant, postgres). A polyrepo would have been "correct" but is overhead for a portfolio project. Instead: one repo, multiple workspace members (`uv` workspaces), and each service has its own `pyproject.toml` and `Dockerfile`. That keeps the build/test surface small while still showing that each service has clear boundaries.

## 2. Why the worker is its own service

Could have just put the ingestion code inside the API. Didn't, because:

- The API process should stay light — Whisper alone can use 1.5 GB resident.
- Ingestion is bursty and CPU-bound; serving is latency-bound. Different SLOs.
- Workers can scale independently and run cron-like tasks (DAG).

## 3. Mock LLM is a feature, not a placeholder

The RAG pipeline ships with a deterministic extractive answerer (`_MockLLM`). This is on purpose:

- CI runs the full Q&A path without API keys.
- Reviewers can clone, `make seed`, and ask questions in < 2 minutes.
- The interface is stable: dropping in OpenAI / Anthropic / Ollama is a single env var change.

## 4. Local Qdrant mode

`qdrant-client` supports an embedded mode that writes to a local folder — no server, no Docker. Used by default. Production uses the real server (`docker-compose.yml`, k8s StatefulSet). The same code path serves both via a `qdrant_mode` setting.

## 5. Mel-spectrogram fallback for audio embeddings

CLAP weights are ~600 MB. On constrained machines or CI, the model load fails. Instead of failing the pipeline, `AudioEmbedder` falls back to a deterministic mel-spectrogram fingerprint with the same dimensionality. Voice matching still works for nearest-neighbor demos; performance is lower but the pipeline is unbroken. The log line `audio_embed.fallback` makes the substitution visible.

## 6. Two-tower over matrix factorization

Pure MF would have been simpler but doesn't support cold-start books. The item tower projects content embeddings (text + audio) into the joint space, so a brand-new book has a real recommendation vector from day one.

## 7. SQLite default, Postgres in prod

Local dev uses SQLite — no service to start, instant. The Postgres URL is one env var swap. Migration story uses Alembic; for now the schema is small enough that `Base.metadata.create_all()` is acceptable.

## 8. Why Streamlit, not Next.js

Initially this was Next.js. Pivoted to Streamlit because:

- Pure Python — one language across the stack signals ML focus.
- ML reviewers expect to see Streamlit; no impedance mismatch.
- 200 lines per page vs 1500 lines for the equivalent React.

The trade-off: Streamlit is harder to make pixel-perfect. For a demo / internal tool / portfolio, that's the right trade.

## 9. Embedding model dimensionality lock

The system stores `text_embed_dim=384` in settings. If the embed model changes (e.g., to `all-mpnet-base-v2` which is 768-dim), every vector has to be re-built. `echomind-worker reembed` does this idempotently.

## 10. What we'd do differently at scale

| Today                          | At Audible scale                                    |
| ------------------------------ | --------------------------------------------------- |
| Polling file watcher           | SQS + S3 events, partitioned by region              |
| In-process Whisper             | Dedicated Triton / TorchServe pool with batching    |
| Single Postgres                | Aurora + read replicas + Redshift for analytics     |
| Local Qdrant                   | Qdrant Cloud or self-hosted 5-node cluster          |
| Synchronous /v1/qa             | Streaming via SSE + LLM token streaming             |
| Sample synthetic interactions  | Real listening events via Kinesis, daily train      |
| Mock LLM in CI                 | Eval vs golden answers, regression on accuracy      |

## Architecture decision records

See `docs/adr/` for individual decisions.
