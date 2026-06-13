# echomind-api

FastAPI service exposing EchoMind's intelligent features.

## Endpoints

| Method | Path                       | Purpose                                |
| ------ | -------------------------- | -------------------------------------- |
| GET    | `/health`                  | Liveness probe                         |
| GET    | `/ready`                   | Readiness — checks DB + Qdrant         |
| GET    | `/metrics`                 | Prometheus metrics                     |
| GET    | `/v1/books`                | List books                             |
| GET    | `/v1/books/{slug}`         | Book detail with chapters              |
| POST   | `/v1/search`               | Semantic search (text / audio / hybrid)|
| POST   | `/v1/qa`                   | RAG-based chapter Q&A                  |
| GET    | `/v1/recommend/user/{id}`  | Personalized recommendations           |
| GET    | `/v1/recommend/book/{id}`  | Similar-book recommendations           |
| POST   | `/v1/voice-match`          | Voice-style nearest neighbor (upload)  |
| GET    | `/v1/sentiment/book/{id}`  | Per-chapter sentiment arc              |
| POST   | `/v1/ingest`               | Trigger ingestion of an audiobook      |

Interactive docs at `/docs` (Swagger) and `/redoc` once running.
