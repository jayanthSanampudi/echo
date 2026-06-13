# EchoMind — Architecture

A pure-Python, modular monorepo. Every box in the diagram below is a separate package or service with its own `pyproject.toml`, and they're stitched together with `uv` workspaces.

## High-level layout

```
                         ┌────────────────────────┐
                         │     Audio sources      │
                         │  (mp3, m4b, wav, ogg)  │
                         └───────────┬────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  echomind-worker    │
                          │  (Typer CLI, DAG)   │
                          └──────────┬──────────┘
                                     │
   ┌─────────────────────────────────┼──────────────────────────────────┐
   │                                 │                                  │
┌──▼───────────┐         ┌───────────▼─────────────┐         ┌──────────▼─────────┐
│ Whisper ASR  │         │  CLAP audio embed       │         │  Sentiment classif │
│ (tiny)       │         │  (or mel fallback)      │         │  (roberta + fallback)│
└──────┬───────┘         └────────────┬────────────┘         └──────────┬─────────┘
       │                              │                                 │
       │ TranscriptSegment[]          │ (dim=512)                       │ score, label
       │                              │                                 │
┌──────▼──────────────────────────────▼─────────────────────────────────▼─────────┐
│                  echomind-core (shared SQL + vector models)                     │
│   ┌──────────────┐    ┌────────────────────────┐    ┌────────────────────────┐ │
│   │  Postgres /  │    │ Qdrant (text & audio)  │    │ Cache (memory/redis)   │ │
│   │  SQLite      │    │  echomind_text/audio   │    │                        │ │
│   └──────────────┘    └────────────────────────┘    └────────────────────────┘ │
└──────────┬──────────────────────────────────────────────────────────────────────┘
           │
   ┌───────▼──────────────────────────────────────────────────────────────┐
   │                          echomind-api (FastAPI)                      │
   │   /v1/search   /v1/qa   /v1/recommend/...   /v1/voice-match          │
   │   /v1/sentiment/...   /v1/books   /v1/ingest   /health  /metrics     │
   └───────┬──────────────────────────────────────────────────────────────┘
           │
   ┌───────▼─────────────────────┐
   │  echomind-ui (Streamlit)    │
   │  multi-page Python app      │
   └─────────────────────────────┘
```

## Workspace packages

| Package           | Role                                                    |
| ----------------- | ------------------------------------------------------- |
| `echomind-core`   | Settings, DB models, Qdrant wrapper, audio utils        |
| `echomind-ml`     | Whisper, embeddings, sentiment, RAG, two-tower model    |
| `echomind-api`    | FastAPI service                                         |
| `echomind-worker` | Typer CLI + Airflow DAG for batch ingestion             |
| `echomind-ui`     | Streamlit UI (talks only to the API — never to the DB)  |

Splitting along these lines means each component can be containerized, scaled, and shipped independently — and `echomind-ml` is reusable in a notebook without dragging the API or UI in.

## Data flow — audio in, intelligence out

1. **Ingest** (`echomind_worker.ingest.ingest_file`):
   - Loads audio to mono 16 kHz with `librosa`.
   - Whisper produces timestamped `TranscriptSegment[]`.
   - Long silences + "Chapter N" matches drive segmentation.
   - Each chapter is sentiment-scored and embedded (text).
   - The first 30 s is embedded into the audio vector space (voice fingerprint).
   - Postgres rows + Qdrant points written in one transaction-friendly path.

2. **Query**:
   - `/v1/search` — embed the query text, k-NN on `echomind_text`.
   - `/v1/qa` — same retrieval, then prompt the LLM with citations.
   - `/v1/voice-match` — embed an uploaded clip, k-NN on `echomind_audio`.
   - `/v1/recommend/user/{handle}` — two-tower lookup, popularity fallback.
   - `/v1/sentiment/book/{slug}` — assemble per-chapter scores from the DB.

3. **Serve**:
   - FastAPI exposes everything with OpenAPI docs at `/docs`.
   - Structured logging via `structlog` (JSON in prod, pretty TTY locally).
   - Prometheus metrics at `/metrics`; HPA in k8s scales on CPU.

## Why these choices

| Choice                | Reason                                                        |
| --------------------- | ------------------------------------------------------------- |
| Whisper-tiny          | 39M params runs CPU-only in real time on a laptop             |
| MiniLM-L6 for text    | 80 MB, 384-dim, surprisingly good for semantic search         |
| CLAP for audio        | Maps audio + text to a shared space — enables voice matching  |
| Qdrant                | Pure Python client, has a local-file mode (no server needed)  |
| Two-tower             | Standard, scalable recommender — same arch as YouTube/TikTok  |
| Streamlit             | Pure Python, the de-facto ML demo UI                          |
| FastAPI               | Async, OpenAPI out of the box, Pydantic for type safety       |
| uv workspaces         | Modern, fast, single-command install for the whole monorepo   |
| Mock LLM fallback     | Lets the RAG pipeline run in CI with no API keys              |
| Mel-spectrogram fallback for CLAP | Same — keeps voice-match working when offline     |

## Scaling notes

- API: stateless, HPA on CPU; vertical scaling helps embedder warmup.
- Worker: long-running batch, scale by sharding the file watcher per node.
- Qdrant: production should use a 3-node cluster with replication.
- Postgres: read replicas for catalog reads, single writer.
- Model cache: ReadWriteMany PVC or download into image layer for fast cold starts.

## What's *not* in scope here

- Live audio streaming (would use WebSockets + chunked Whisper).
- Federated personalization (sketched in the roadmap).
- Multi-tenant auth — the API is unauthenticated for the demo.
