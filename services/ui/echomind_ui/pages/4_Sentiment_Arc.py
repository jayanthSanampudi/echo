"""Sentiment-arc visualization."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from echomind_ui.api_client import get_client

st.set_page_config(page_title="Sentiment Arc — EchoMind", page_icon="📈", layout="wide")
st.title("Sentiment Arc")
st.caption("How does the emotional tone shift across the book?")

client = get_client()

try:
    books = client.list_books(limit=200)
except Exception as e:
    st.error(f"could not load catalog: {e}")
    st.stop()

if not books:
    st.info("No books indexed yet.")
    st.stop()

book_options = {f"{b['title']} ({b['slug']})": b for b in books}
choice = st.selectbox("Book", list(book_options.keys()))
book = book_options[choice]

try:
    arc = client.sentiment_arc(book["slug"])
except Exception as e:
    st.error(f"could not load arc: {e}")
    st.stop()

points = arc.get("points", [])
if not points:
    st.info("No sentiment data computed for this book yet.")
    st.stop()

df = pd.DataFrame(points)
df["chapter"] = df["chapter_idx"] + 1

fig = px.line(
    df,
    x="chapter",
    y="score",
    markers=True,
    title=f"Sentiment per chapter — {book['title']}",
    labels={"score": "sentiment (-1 to +1)", "chapter": "chapter"},
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Raw chapter data"):
    st.dataframe(df[["chapter", "start_sec", "end_sec", "label", "score"]])
