"""EchoMind Streamlit — home page.

Streamlit multi-page app: the `pages/` directory next to this file holds each
feature. Run with: `streamlit run services/ui/echomind_ui/app.py`.
"""

from __future__ import annotations

import streamlit as st

from echomind_ui.api_client import get_client

st.set_page_config(
    page_title="EchoMind",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EchoMind")
st.caption("AI-powered audiobook intelligence — search, recommend, converse.")

client = get_client()
is_healthy = client.health()

col1, col2, col3 = st.columns(3)
col1.metric("API status", "online" if is_healthy else "offline")
try:
    books = client.list_books(limit=200)
    col2.metric("Books indexed", len(books))
    total_chapters = sum(len(b.get("chapters", [])) for b in books)
    col3.metric("Chapters", total_chapters)
except Exception:
    col2.metric("Books indexed", "—")
    col3.metric("Chapters", "—")

st.divider()

st.markdown(
    """
### What you can do

- **Search** — semantic search across every chapter of every book.
- **Q&A** — ask a question about a book, get an answer with citations.
- **Voice Match** — upload an audio clip, find books with similar narration.
- **Sentiment Arc** — see a book's emotional trajectory over its chapters.
- **Recommend** — personalized + similar-book recommendations.
- **Upload** — ingest a new audiobook into the catalog.

Pick a feature from the sidebar to get started.
"""
)

if not is_healthy:
    st.error(
        "The EchoMind API is not reachable. Start it with `make api` or "
        "`uv run uvicorn echomind_api.main:app --port 8000`."
    )
