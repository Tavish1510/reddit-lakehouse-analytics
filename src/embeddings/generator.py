import chromadb
from chromadb.config import Settings as ChromaSettings
from pyspark.sql import DataFrame
from sentence_transformers import SentenceTransformer

from config import settings


def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.CHROMA_PATH)


def build_vector_store(df: DataFrame, batch_size: int = 256):
    model = get_embedding_model()
    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name="reddit_posts",
        metadata={"hnsw:space": "cosine"},
    )

    rows = df.select("post_id", "subreddit", "title", "content").collect()

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [row["content"][:2000] for row in batch]
        ids = [row["post_id"] for row in batch]
        metadatas = [
            {"subreddit": row["subreddit"], "title": row["title"]}
            for row in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"Embedded {min(i + batch_size, len(rows))}/{len(rows)} posts")

    print(f"Vector store built with {collection.count()} documents at {settings.CHROMA_PATH}")
    return collection


def search(query: str, n_results: int = 5, subreddit_filter: str | None = None) -> list[dict]:
    model = get_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection("reddit_posts")

    query_embedding = model.encode([query]).tolist()

    search_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if subreddit_filter:
        search_kwargs["where"] = {"subreddit": subreddit_filter}

    results = collection.query(**search_kwargs)

    posts = []
    for i in range(len(results["ids"][0])):
        posts.append(
            {
                "post_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "subreddit": results["metadatas"][0][i]["subreddit"],
                "title": results["metadatas"][0][i]["title"],
                "distance": results["distances"][0][i],
            }
        )

    return posts
