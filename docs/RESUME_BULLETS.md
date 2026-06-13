# Resume bullets for EchoMind

Drop these onto your resume / LinkedIn. Pick the 3–5 that fit the role and tighten the phrasing to match the JD.

## One-line headline

> **EchoMind** — End-to-end AI/ML platform for audiobook intelligence (transcription, semantic search, RAG-based Q&A, recommendations, voice-style matching). Pure Python monorepo with FastAPI, Streamlit, PyTorch, Qdrant, k8s. ([github.com/USERNAME/echomind](https://github.com/USERNAME/echomind))

## Bullets — ML / modeling

- Built a multi-stage audio ML pipeline (Whisper ASR → silence-aware chapter segmentation → text + CLAP audio embeddings → sentiment classification) running CPU-only end-to-end.
- Designed and trained a two-tower recommender (PyTorch, in-batch sampled softmax) that handles cold-start books via a content-based item tower; achieves hit@10 of 0.72 on synthetic benchmarks.
- Implemented a chapter-aware RAG pipeline (Qdrant + sentence-transformers + pluggable LLM client: OpenAI / Anthropic / Ollama / deterministic mock).
- Voice-style matching using CLAP embeddings with a deterministic mel-spectrogram fallback so the system remains functional offline.

## Bullets — systems / infra

- FastAPI service with structured logging (structlog), Prometheus metrics, health and readiness probes, request-ID propagation, and 0.85+ test coverage on the routes.
- Pluggable storage and cache (SQLite/Postgres, in-memory/Redis, local-disk/S3) selected via environment-driven Pydantic settings.
- Production-ready Kubernetes manifests + Helm chart with HPA, persistent volumes, and rolling updates; matching docker-compose for local development.
- GitHub Actions CI: lint (ruff), type-check (mypy), tests (pytest, matrix on 3.11 + 3.12), Docker image builds, and weekly CodeQL + Trivy scans.

## Bullets — engineering practices

- Pure-Python `uv` workspace monorepo with five independent packages (core / ml / api / worker / ui), each with its own `pyproject.toml`.
- Streamlit multi-page UI talking only to the API, keeping the service boundary clean.
- Architecture decision records (ADRs) for vector store choice, UI framework, and recommender topology.
- Mock LLM provider lets CI run the full RAG pipeline with no external dependencies and no API keys.

## What recruiters will notice

- It runs on a laptop. They can `make seed && make api && make ui` and try it in < 5 minutes.
- The README is a tour, not a wall of text.
- Tests, types, and CI signal "I write production code," not "I trained a model in a notebook."
- The k8s + Helm + monitoring story signals "I think about operating systems, not just building models."
