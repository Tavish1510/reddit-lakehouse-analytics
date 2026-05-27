import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

from config import settings
from sentence_transformers import SentenceTransformer


class SemanticSearchClient:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.collection = self.client.get_collection("reddit_posts")

    def search(self, query: str, n_results: int = 5, subreddit_filter: str | None = None) -> list[dict]:
        query_embedding = self.model.encode([query]).tolist()

        search_kwargs = {
            "query_embeddings": query_embedding,
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if subreddit_filter:
            search_kwargs["where"] = {"subreddit": subreddit_filter}

        results = self.collection.query(**search_kwargs)

        posts = []
        for i in range(len(results["ids"][0])):
            posts.append(
                {
                    "post_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "subreddit": results["metadatas"][0][i]["subreddit"],
                    "title": results["metadatas"][0][i]["title"],
                    "similarity": 1 - results["distances"][0][i],
                }
            )

        return posts
