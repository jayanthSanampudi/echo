# echomind-worker

The batch ingestion worker. Takes a raw audio file and runs it through the full ML pipeline:

```
audio.mp3
   │
   ▼
Whisper ASR ─► transcript segments (start, end, text)
   │
   ├─► chapter segmentation (silence + transcript heuristic)
   │      └─► persist chapters to DB
   │
   ├─► per-chapter sentiment ─► persist score + label
   │
   ├─► text embeddings (chapter-level) ─► Qdrant (echomind_text)
   │
   └─► audio embedding (book-level CLAP) ─► Qdrant (echomind_audio)
```

## Usage

```bash
# ingest a single file
uv run echomind-worker ingest data/audio/frankenstein.mp3 \
    --slug frankenstein --title "Frankenstein" --author "Mary Shelley"

# watch a directory for new files
uv run echomind-worker watch data/audio

# re-embed everything (e.g. after changing the embedding model)
uv run echomind-worker reembed
```
