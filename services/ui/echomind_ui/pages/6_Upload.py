"""Trigger ingestion of a new audiobook."""

from __future__ import annotations

import streamlit as st

from echomind_ui.api_client import get_client

st.set_page_config(page_title="Upload — EchoMind", page_icon="⬆️", layout="wide")
st.title("Upload an audiobook")
st.caption(
    "Register metadata here, then drop the file into `data/audio/` "
    "and the worker will pick it up. Or run the CLI: "
    "`uv run echomind-worker ingest path/to/audio.mp3 ...`."
)

client = get_client()

with st.form("ingest_form"):
    slug = st.text_input("Slug *", placeholder="frankenstein", help="URL-safe identifier")
    title = st.text_input("Title *", placeholder="Frankenstein")
    col1, col2 = st.columns(2)
    author = col1.text_input("Author", placeholder="Mary Shelley")
    narrator = col2.text_input("Narrator", placeholder="LibriVox volunteers")
    col3, col4 = st.columns(2)
    genre = col3.text_input("Genre", placeholder="gothic-horror")
    language = col4.text_input("Language", value="en")
    audio_url = st.text_input("Audio URL (optional)")

    submit = st.form_submit_button("Queue ingestion", type="primary")

if submit:
    if not slug or not title:
        st.error("slug and title are required")
        st.stop()
    try:
        result = client.ingest(
            {
                "slug": slug,
                "title": title,
                "author": author or None,
                "narrator": narrator or None,
                "genre": genre or None,
                "language": language or "en",
                "audio_url": audio_url or None,
            }
        )
    except Exception as e:
        st.error(f"ingest failed: {e}")
        st.stop()

    st.success(f"queued — book_id `{result['book_id']}` ({result['status']})")
    st.code(
        f"# now run:\n"
        f"uv run echomind-worker ingest data/audio/{slug}.mp3 \\\n"
        f"    --slug {slug} \\\n"
        f"    --title \"{title}\""
    )
