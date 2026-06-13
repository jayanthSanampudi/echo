<div align="center">

# EchoMind

**An end-to-end AI/ML platform for audiobook intelligence.**

Transcribe → embed → search → recommend → converse — all in one pluggable pipeline.

[![CI](https://github.com/USERNAME/echomind/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/echomind/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## What it is

EchoMind ingests audiobooks and podcasts, runs them through a multi-stage ML pipeline (speech recognition → speaker diarization → text/audio embeddings → chapter segmentation → sentiment analysis), and exposes intelligent features over the indexed content:

- **Semantic search** over the full audio catalog (`"find suspenseful sci-fi narrated by a deep voice"`)
- **Personalized recommendations** via a two-tower neural model
- **Voice-style matching** — find narrators that sound similar to a sample
- **Chapter-aware Q&A** with retrieval-augmented generation
- **Emotion-arc visualization** of a book across chapters
- **Automatic chapter segmentation** from raw audio

The whole system runs on a laptop (CPU-only, no GPU needed), but every component is designed to scale — vector DB, message queue, k8s manifests, drift monitoring, and an A/B framework are all in the repo.

## Why this project exists

I built EchoMind as a portfolio piece while preparing for AI/ML SDE roles. The goal was to demonstrate every step of the production ML lifecycle in one focused domain (audio storytelling):

| ML lifecycle stage      | Where in this repo                                                              |
| ----------------------- | ------------------------------------------------------------------------------- |
| Data ingestion          | `pipelines/ingest_dag.py`, `scripts/seed_data.py`                               |
| Model training          | `notebooks/03_recommender_training.ipynb`, `packages/ml/echomind_ml/recommender.py` |
| Model inference         | `packages/ml/echomind_ml/` (ASR, embeddings, RAG)                               |
| Serving                 | `services/api/` (FastAPI, sub-100ms p95 for search)                             |
| Evaluation              | `notebooks/02_embeddings_eval.ipynb`, `scripts/benchmark.py`                    |
| Monitoring              | `infra/k8s/`, Prometheus metrics in API                                         |
| Frontend                | `services/frontend/` (Next.js)                                                  |
| Tests + CI/CD           | `tests/`, `.github/workflows/`                                                  |

## Architecture

```
                          ┌────────────────┐
                          │  Audio source  │
                          │ (mp3, m4b, wav)│
                          └────────┬───────┘
                                   │
                          ┌────────▼────────┐
                          │  Ingest worker  │
                          │   (CLI / DAG)   │
                          └────────┬────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
       ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐
       │ Whisper ASR │      │ CLAP audio  │      │  Sentiment  │
       │ (tiny/base) │      │  embedder   │      │  classifier │
       └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
              │                    │                    │
       ┌──────▼─────────────────────▼────────────────────▼──────┐
       │  Postgres / SQLite  +  Qdrant (vector DB)              │
       │  (transcripts, metadata, sentiment arcs, embeddings)   │
       └─────────────────────────┬──────────────────────────────┘
                                 │
                       ┌─────────▼──────────┐
                       │   FastAPI service  │
                       │  /search /qa /rec  │
                       │  /voice  /sentiment│
                       └─────────┬──────────┘
                                 │
                       ┌─────────▼──────────┐
                       │   Streamlit UI     │
                       └────────────────────┘
```

For a deeper dive, see [`docs/architecture.md`](docs/architecture.md) and [`docs/design.md`](docs/design.md).

## Quickstart

### Option A — native (no Docker)

```bash
git clone https://github.com/USERNAME/echomind.git
cd echomind

# install uv if you don't have it: https://docs.astral.sh/uv/
uv sync

# seed a tiny LibriVox sample (≈2 min audio)
uv run python scripts/seed_data.py

# run the API
uv run uvicorn echomind_api.main:app --reload --port 8000

# in another shell, run the Streamlit UI
uv run streamlit run services/ui/echomind_ui/app.py
```

Visit `http://localhost:8501`.

### Option B — Docker Compose

```bash
docker compose up --build
```

That brings up the API, worker, Streamlit UI, Qdrant, Postgres, and Redis.

## API examples

```bash
# Semantic search
curl -X POST localhost:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "a tense sci-fi chapter about first contact", "k": 5}'

# Chapter Q&A (RAG)
curl -X POST localhost:8000/v1/qa \
  -H 'Content-Type: application/json' \
  -d '{"book_id": "frankenstein", "question": "Who does Victor meet on the glacier?"}'

# Personalized recommendations
curl localhost:8000/v1/recommend/user/u_42?k=10

# Voice-style match (upload audio)
curl -X POST localhost:8000/v1/voice-match \
  -F 'sample=@my_narrator_sample.wav'

# Sentiment arc
curl localhost:8000/v1/sentiment/book/frankenstein
```

Full OpenAPI docs at `http://localhost:8000/docs` once running.

## Features

### 1. Multi-modal embeddings

Text and audio are embedded into separate but compatible vector spaces. Text uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~80MB), audio uses `laion/clap-htsat-unfused` (512-dim). The search endpoint can query either or both, with a configurable hybrid score.

### 2. Two-tower recommender

A PyTorch model (`packages/ml/echomind_ml/recommender.py`) jointly embeds users (from listening history) and books (from content embeddings + metadata). Trained on synthetic interactions derived from the LibriVox catalog. Evaluation uses hit@k and nDCG; baseline metrics are in `notebooks/03_recommender_training.ipynb`.

### 3. Chapter-aware RAG

Each book's transcript is chunked at semantic boundaries (chapter / paragraph), embedded, and indexed. The `/qa` endpoint retrieves the top-k chunks and prompts an LLM (configurable — OpenAI-compatible or local) to answer with citations.

### 4. Voice-style matching

A novel use of CLAP — embed a narrator audio sample, nearest-neighbor lookup against indexed audio embeddings, return books with similar narration style.

### 5. Sentiment arc

Per-chapter sentiment scores are computed during ingestion and exposed as a time series, so the UI can render the emotional trajectory of a book.

### 6. Chapter segmentation

Hybrid approach — silence detection (`librosa`) provides candidate boundaries, then a lightweight classifier (`packages/ml/echomind_ml/chapter_seg.py`) confirms whether each is a chapter break.

### 7. Speaker diarization

Identifies distinct speakers in multi-narrator content. Useful for podcasts and audiobooks with character voices.

## Tech stack

**ML & Data**
PyTorch · HuggingFace Transformers · OpenAI Whisper · CLAP · sentence-transformers · scikit-learn · librosa

**Backend**
FastAPI · Pydantic v2 · SQLAlchemy 2 · Qdrant (vector DB) · Redis (cache) · Postgres / SQLite

**Frontend**
Streamlit (pure Python, multi-page) · Plotly · Altair

**MLOps & Infra**
MLflow · DVC · Evidently (drift) · Prometheus + Grafana · Docker · k8s · Helm · GitHub Actions

**DevEx**
uv · ruff · mypy · pytest · pre-commit

## Project structure

```
echomind/
├── packages/
│   ├── core/                 # Shared lib: DB, vector, audio utils
│   └── ml/                   # Model wrappers: ASR, embed, RAG, recommender
├── services/
│   ├── api/                  # FastAPI service
│   ├── worker/               # Batch ingestion worker
│   └── ui/                   # Streamlit UI (pure Python)
├── pipelines/                # Airflow / Prefect DAGs
├── scripts/                  # Seed data, benchmarks
├── tests/                    # pytest suite
├── notebooks/                # EDA + eval
├── docs/                     # Architecture, design, ADRs
├── infra/
│   ├── k8s/                  # Raw k8s manifests
│   └── helm/                 # Helm chart
├── data/                     # Sample audio + transcripts (gitignored beyond samples)
└── .github/workflows/        # CI/CD
```

## Development

```bash
# install all packages in editable mode
uv sync

# run tests
uv run pytest

# lint + type check
uv run ruff check .
uv run mypy packages services

# run a single feature locally
uv run python -m echomind_worker.transcribe data/audio/sample.mp3

# train the recommender on synthetic data
uv run python -m echomind_ml.recommender --train
```

## Benchmarks (CPU, M1 / Intel laptops)

| Operation                              | Latency (p95) | Notes                                     |
| -------------------------------------- | ------------: | ----------------------------------------- |
| Text embedding (single query)          |         18 ms | MiniLM-L6                                 |
| Vector search (10k items, k=20)        |          6 ms | Qdrant local                              |
| End-to-end search (`/search`)          |         34 ms | warm cache                                |
| Whisper-tiny transcription             |    ~0.3× RT   | 1 min audio → ~18s transcription on CPU   |
| RAG `/qa` (5 chunks + LLM)             |     ~1.4 s    | with local Phi-3-mini or OpenAI mock      |

See `scripts/benchmark.py` to reproduce.

## Roadmap

- [ ] Fine-tune Whisper on narrator-style domain
- [ ] Real-time streaming transcription via WebSocket
- [ ] Add a small TTS demo (Bark / Coqui XTTS)
- [ ] Production A/B framework with traffic split
- [ ] Multi-language support (Whisper handles many — needs UI work)
- [ ] Federated training experiment for personalization

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, conventions, and the PR checklist.

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

Built with [Whisper](https://github.com/openai/whisper), [HuggingFace Transformers](https://github.com/huggingface/transformers), [Qdrant](https://github.com/qdrant/qdrant), [FastAPI](https://github.com/tiangolo/fastapi), and [Next.js](https://github.com/vercel/next.js). Inspired by working on audio-first ML problems and a love for audiobooks.
