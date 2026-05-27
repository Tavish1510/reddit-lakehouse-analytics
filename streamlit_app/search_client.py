import os

from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI


class SemanticSearchClient:
    def __init__(self, databricks_host: str, databricks_token: str, openai_api_key: str):
        os.environ["DATABRICKS_HOST"] = databricks_host
        os.environ["DATABRICKS_TOKEN"] = databricks_token

        self.vsc = VectorSearchClient(
            workspace_url=databricks_host,
            personal_access_token=databricks_token,
        )
        self.openai = OpenAI(api_key=openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        self.vs_endpoint = "reddit_vs_endpoint"
        self.vs_index = "reddit_analytics.gold.post_embeddings_index"

    def get_query_embedding(self, query: str) -> list[float]:
        response = self.openai.embeddings.create(
            model=self.embedding_model, input=[query]
        )
        return response.data[0].embedding

    def search(self, query: str, num_results: int = 5, filters: dict | None = None) -> list[dict]:
        query_embedding = self.get_query_embedding(query)

        search_kwargs = {
            "query_vector": query_embedding,
            "columns": ["post_id", "subreddit", "title", "content"],
            "num_results": num_results,
        }
        if filters:
            search_kwargs["filters"] = filters

        index = self.vsc.get_index(self.vs_endpoint, self.vs_index)
        results = index.similarity_search(**search_kwargs)

        posts = []
        for row in results.get("result", {}).get("data_array", []):
            posts.append({
                "post_id": row[0],
                "subreddit": row[1],
                "title": row[2],
                "content": row[3],
                "similarity_score": row[-1],
            })

        return posts

    def answer_question(self, query: str, num_results: int = 5) -> dict:
        relevant_posts = self.search(query, num_results=num_results)

        context = "\n\n---\n\n".join(
            f"[r/{p['subreddit']}] {p['title']}\n{p['content'][:500]}"
            for p in relevant_posts
        )

        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions based on Reddit posts. "
                        "Use the provided context to answer the user's question. "
                        "Cite the subreddit and post title when referencing specific information. "
                        "If the context doesn't contain relevant information, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context from Reddit posts:\n\n{context}\n\nQuestion: {query}",
                },
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": relevant_posts,
        }
