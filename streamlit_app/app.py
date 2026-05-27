import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from search_client import SemanticSearchClient
from config import settings

st.set_page_config(page_title="Reddit Lakehouse Q&A", page_icon="🔍", layout="wide")

st.title("Reddit Lakehouse Q&A")
st.caption("Semantic search over Reddit posts using local embeddings (Sentence-Transformers + ChromaDB)")


@st.cache_resource
def get_client():
    return SemanticSearchClient()


client = get_client()

SUBREDDITS = ["All"] + settings.SUBREDDITS

with st.sidebar:
    st.header("Settings")
    subreddit_filter = st.selectbox("Filter by Subreddit", SUBREDDITS)
    num_results = st.slider("Number of results", 3, 20, 5)

query = st.text_input(
    "Search Reddit posts:",
    placeholder="e.g., What do people think about learning Python in 2025?",
)

if query:
    sub_filter = None if subreddit_filter == "All" else subreddit_filter

    with st.spinner("Searching..."):
        results = client.search(query, n_results=num_results, subreddit_filter=sub_filter)

    st.subheader(f"{len(results)} Results")

    for i, post in enumerate(results, 1):
        similarity_pct = post["similarity"] * 100
        with st.expander(f"{i}. [r/{post['subreddit']}] {post['title']} — {similarity_pct:.1f}% match"):
            st.markdown(post["content"][:1500])
            st.markdown(f"[View on Reddit](https://reddit.com/comments/{post['post_id']})")
