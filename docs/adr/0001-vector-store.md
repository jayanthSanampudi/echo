# ADR 0001 — Vector store choice

- Status: Accepted
- Date: 2026-04-15

## Context

EchoMind needs a vector store for two purposes:

1. Semantic search over chapter text (~384-dim embeddings, 10k–100k chunks).
2. Voice-style nearest-neighbor over CLAP audio embeddings (~512-dim).

It must work both (a) locally on a developer laptop with no extra services, and (b) in production at higher scale.

## Options considered

| Option            | Pros                                              | Cons                                                     |
| ----------------- | ------------------------------------------------- | -------------------------------------------------------- |
| FAISS             | Fast, in-process, well-known                      | No payload filters → need a sidecar metadata store       |
| pgvector          | One DB to operate                                 | Slower for high-dim; payload schemas are SQL columns     |
| Pinecone          | Managed, scalable                                 | External SaaS, no local mode, costs                      |
| Weaviate          | Native filters + multi-modal                      | Heavier to run locally                                   |
| **Qdrant**        | Local file mode, fast filters, Python-native, OSS | Need a separate service in prod (acceptable)             |

## Decision

Use **Qdrant**, with `QDRANT_MODE=local` (file-backed) for dev and a Qdrant server for production.

## Consequences

- Zero-friction local dev — no Docker required just to run a query.
- Payload filters (e.g. `book_id=X`) move out of SQL and live next to vectors. We mirror only the minimum payload into Qdrant; the SQL layer remains source of truth.
- A migration path to a managed Qdrant cluster is straightforward — same client, different URL.
