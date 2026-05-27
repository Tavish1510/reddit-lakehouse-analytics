import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

from search_client import SemanticSearchClient

st.set_page_config(
    page_title="Reddit Lakehouse Q&A",
    page_icon="🔍",
    layout="wide",
)

st.title("Reddit Lakehouse Q&A")
st.caption("Semantic search over Reddit posts using RAG (Retrieval-Augmented Generation)")


@st.cache_resource
def get_client():
    return SemanticSearchClient(
        databricks_host=os.getenv("DATABRICKS_HOST"),
        databricks_token=os.getenv("DATABRICKS_TOKEN"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


client = get_client()

SUBREDDITS = [
    "All",
    "technology", "programming", "MachineLearning", "datascience",
    "wallstreetbets", "personalfinance", "investing",
    "science", "space",
    "worldnews", "news",
    "AskReddit", "explainlikeimfive",
    "fitness", "cooking", "travel",
]

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Q&A (AI Answer)", "Search Only"])
    subreddit_filter = st.selectbox("Filter by Subreddit", SUBREDDITS)
    num_results = st.slider("Number of results", 3, 20, 5)

query = st.text_input("Ask a question or search Reddit posts:", placeholder="e.g., What do people think about learning Python in 2025?")

if query:
    filters = None
    if subreddit_filter != "All":
        filters = {"subreddit": subreddit_filter}

    with st.spinner("Searching..."):
        if mode == "Q&A (AI Answer)":
            result = client.answer_question(query, num_results=num_results)

            st.subheader("Answer")
            st.markdown(result["answer"])

            st.subheader("Sources")
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"{i}. [r/{source['subreddit']}] {source['title']}"):
                    st.markdown(f"**Similarity Score:** {source['similarity_score']:.4f}")
                    st.markdown(source["content"][:1000])
                    st.markdown(f"[View on Reddit](https://reddit.com/comments/{source['post_id']})")
        else:
            results = client.search(query, num_results=num_results, filters=filters)

            st.subheader(f"{len(results)} Results")
            for i, post in enumerate(results, 1):
                with st.expander(f"{i}. [r/{post['subreddit']}] {post['title']} (score: {post['similarity_score']:.4f})"):
                    st.markdown(post["content"][:1000])
                    st.markdown(f"[View on Reddit](https://reddit.com/comments/{post['post_id']})")
