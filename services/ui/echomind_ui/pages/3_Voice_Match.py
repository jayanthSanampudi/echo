"""Voice-style matching page."""

from __future__ import annotations

import streamlit as st

from echomind_ui.api_client import get_client

st.set_page_config(page_title="Voice Match — EchoMind", page_icon="🎙️", layout="wide")
st.title("Voice-Style Matching")
st.caption("Upload an audio clip — get books with similar narration style.")

client = get_client()

uploaded = st.file_uploader(
    "Audio sample (wav / mp3 / m4a, ≤25MB)", type=["wav", "mp3", "m4a", "flac", "ogg"]
)
k = st.slider("Top-K matches", min_value=3, max_value=20, value=10)

if uploaded is not None:
    st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
    if st.button("Find similar voices", type="primary"):
        with st.spinner("Embedding and searching..."):
            try:
                data = uploaded.getvalue()
                result = client.voice_match(data, uploaded.name, k=k)
            except Exception as e:
                st.error(f"voice match failed: {e}")
                st.stop()

        hits = result.get("hits", [])
        if not hits:
            st.info("No matches found. Ingest more audiobooks via **Upload**.")
        else:
            st.success(f"Top {len(hits)} matches")
            for h in hits:
                with st.container(border=True):
                    st.markdown(f"**{h['title']}** — score `{h['score']:.3f}`")
                    if h.get("narrator"):
                        st.caption(f"narrator: {h['narrator']}")
