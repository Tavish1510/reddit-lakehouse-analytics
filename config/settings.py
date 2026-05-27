import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BRONZE_PATH = str(DATA_DIR / "bronze" / "raw_posts")
SILVER_PATH = str(DATA_DIR / "silver" / "cleaned_posts")
GOLD_DIR = str(DATA_DIR / "gold")
CHROMA_PATH = str(DATA_DIR / "chroma_db")

REDDIT_USER_AGENT = "RedditLakehouse/1.0 (educational project)"
REDDIT_BASE_URL = "https://www.reddit.com/r/{subreddit}/{sort}.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

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
