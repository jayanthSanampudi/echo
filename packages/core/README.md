# echomind-core

Shared library for EchoMind. Houses cross-cutting concerns:

- `config` — typed settings via `pydantic-settings`
- `models` — SQLAlchemy ORM models
- `schemas` — Pydantic schemas (API + internal)
- `db` — engine and session helpers
- `vector` — Qdrant client (local-file or server mode)
- `cache` — in-memory / Redis cache backends
- `storage` — local-disk / S3 file storage
- `audio` — load, resample, chunk audio files
- `logging` — `structlog` configuration

Importable from anywhere in the monorepo as `echomind_core`.
