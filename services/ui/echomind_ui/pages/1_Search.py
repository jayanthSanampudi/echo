"""Semantic search page."""

from __future__ import annotations

import streamlit as st

from echomind_ui.api_client import get_client

st.set_page_config(page_title="Search — EchoMind", page_icon="🔎", layout="wide")
st.title("Semantic Search")
st.caption("Find chapters by meaning, not just keywords.")

client = get_client()

with st.form("search_form"):
    query = st.text_input("Query", placeholder="a tense sci-fi chapter about first contact")
    col1, col2, col3 = st.columns(3)
    k = col1.slider("Results", min_value=3, max_value=30, value=10)
    mode = col2.selectbox("Mode", ["text", "audio", "hybrid"], index=0)
    genre = col3.text_input("Genre filter (optional)")
    submitted = st.form_submit_button("Search", type="primary")

if submitted and query.strip():
    with st.spinner("Searching..."):
        try:
            results = client.search(query=query, k=k, mode=mode, genre=genre or None)
        except Exception as e:
            st.error(f"search failed: {e}")
            st.stop()

    st.success(f"Found {len(results['hits'])} results in {results['latency_ms']:.1f} ms")
    for i, hit in enumerate(results["hits"], 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {hit.get('title', 'Untitled')}** — score `{hit['score']:.3f}`")
            if hit.get("author"):
                st.caption(f"by {hit['author']}")
            st.write(hit["snippet"])
