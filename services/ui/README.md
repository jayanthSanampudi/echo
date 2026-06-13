# echomind-ui

A pure-Python Streamlit multi-page app that drives every EchoMind feature through the API.

Pages:

1. **Home** — overview + system stats
2. **Search** — semantic search with filters
3. **Q&A** — ask questions about a book (RAG)
4. **Voice Match** — upload an audio sample, find similar narrators
5. **Sentiment Arc** — visualize a book's emotional trajectory
6. **Recommend** — per-user and similar-book recommendations
7. **Upload** — trigger ingestion of a new audiobook

Run it:

```bash
uv run streamlit run services/ui/echomind_ui/app.py
```
