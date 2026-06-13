"""Recommendations page."""

from __future__ import annotations

import streamlit as st

from echomind_ui.api_client import get_client

st.set_page_config(page_title="Recommend — EchoMind", page_icon="✨", layout="wide")
st.title("Recommendations")

client = get_client()

tab1, tab2 = st.tabs(["For a user", "Similar to a book"])

with tab1:
    handle = st.text_input("User handle", value="alex", help="defaults to popularity if user is unknown")
    k = st.slider("How many?", 1, 30, 10, key="user_k")
    if st.button("Recommend", type="primary", key="user_btn"):
        try:
            result = client.recommend_user(handle, k=k)
        except Exception as e:
            st.error(f"failed: {e}")
            st.stop()

        for b, why in zip(result["books"], result["explain"] + [""] * len(result["books"])):
            with st.container(border=True):
                st.markdown(f"**{b['title']}**" + (f" by {b['author']}" if b.get("author") else ""))
                if why:
                    st.caption(why)

with tab2:
    try:
        books = client.list_books(limit=200)
    except Exception as e:
        st.error(f"could not load catalog: {e}")
        st.stop()

    if not books:
        st.info("No books indexed yet.")
    else:
        book_options = {f"{b['title']} ({b['slug']})": b for b in books}
        choice = st.selectbox("Pick a book", list(book_options.keys()), key="sim_book")
        seed_book = book_options[choice]
        k2 = st.slider("How many?", 1, 30, 10, key="book_k")
        if st.button("Find similar", type="primary", key="book_btn"):
            try:
                result = client.recommend_book(seed_book["id"], k=k2)
            except Exception as e:
                st.error(f"failed: {e}")
                st.stop()

            for b in result["books"]:
                with st.container(border=True):
                    st.markdown(f"**{b['title']}**" + (f" by {b['author']}" if b.get("author") else ""))
