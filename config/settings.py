import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "RedditLakehouse/1.0")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSION = 1536

    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

    CATALOG = "reddit_analytics"
    SCHEMA_BRONZE = "bronze"
    SCHEMA_SILVER = "silver"
    SCHEMA_GOLD = "gold"

    VECTOR_SEARCH_ENDPOINT = "reddit_vs_endpoint"
    VECTOR_SEARCH_INDEX = f"{CATALOG}.gold.post_embeddings_index"

    SUBREDDITS = [
        "technology",
        "programming",
        "MachineLearning",
        "datascience",
        "wallstreetbets",
        "personalfinance",
        "investing",
        "science",
        "space",
        "worldnews",
        "news",
        "AskReddit",
        "explainlikeimfive",
        "fitness",
        "cooking",
        "travel",
    ]

    POSTS_PER_SUBREDDIT = 500
