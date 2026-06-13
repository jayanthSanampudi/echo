"""Chapter Q&A page (RAG)."""

from __future__ import annotations

import streamlit as st

from echomind_ui.api_client import get_client

st.set_page_config(page_title="Q&A — EchoMind", page_icon="💬", layout="wide")
st.title("Ask a question about a book")
st.caption("Retrieval-augmented generation over the book's transcript.")

client = get_client()

try:
    books = client.list_books(limit=200)
except Exception as e:
    st.error(f"could not load catalog: {e}")
    st.stop()

if not books:
    st.info("No books indexed yet. Use the **Upload** page or run `make seed`.")
    st.stop()

book_options = {f"{b['title']} ({b['slug']})": b for b in books}
choice = st.selectbox("Book", list(book_options.keys()))
book = book_options[choice]

question = st.text_area(
    "Question", placeholder="What happens to Victor on the glacier?", height=100
)
k = st.slider("Number of context chunks", 1, 15, 5)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Thinking..."):
        try:
            result = client.qa(book_id=book["id"], question=question, k=k)
        except Exception as e:
            st.error(f"qa failed: {e}")
            st.stop()

    st.markdown("### Answer")
    st.write(result["answer"])
    st.caption(f"Latency: {result['latency_ms']:.1f} ms")

    with st.expander(f"Citations ({len(result['citations'])})", expanded=False):
        for c in result["citations"]:
            st.markdown(
                f"**Chapter {c['chapter_idx']}** "
                f"({c['start_sec']:.0f}s – {c['end_sec']:.0f}s)"
            )
            st.caption(c["text"])
            st.divider()
