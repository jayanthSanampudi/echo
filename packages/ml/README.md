# echomind-ml

Model wrappers and ML primitives.

| Module           | What it does                                                    |
| ---------------- | --------------------------------------------------------------- |
| `asr`            | OpenAI Whisper transcription (tiny by default — CPU friendly)   |
| `embed_text`     | Sentence-transformer text embeddings (`MiniLM-L6-v2`)           |
| `embed_audio`    | CLAP audio embeddings                                           |
| `sentiment`      | RoBERTa sentiment classifier                                    |
| `chapter_seg`    | Hybrid silence + heuristic chapter segmentation                 |
| `diarize`        | Lightweight speaker-turn detection                              |
| `recommender`    | Two-tower PyTorch recommender (user × content)                  |
| `llm`            | LLM client abstraction (OpenAI / Anthropic / Ollama / mock)     |
| `rag`            | Retrieval-augmented generation pipeline for chapter Q&A         |

All models lazy-load and cache under `MODEL_CACHE_DIR`. The "mock" LLM lets the whole RAG pipeline run with zero API keys, which means CI tests and `make seed` work offline.
