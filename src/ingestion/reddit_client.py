import time
from datetime import datetime, timezone

import requests
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from config import settings

POST_SCHEMA = StructType(
    [
        StructField("post_id", StringType(), False),
        StructField("subreddit", StringType(), False),
        StructField("title", StringType(), False),
        StructField("selftext", StringType(), True),
        StructField("author", StringType(), True),
        StructField("score", IntegerType(), True),
        StructField("upvote_ratio", FloatType(), True),
        StructField("num_comments", IntegerType(), True),
        StructField("created_utc", TimestampType(), False),
        StructField("url", StringType(), True),
        StructField("permalink", StringType(), True),
        StructField("is_self", StringType(), True),
        StructField("link_flair_text", StringType(), True),
        StructField("over_18", StringType(), True),
        StructField("spoiler", StringType(), True),
        StructField("awards_received", IntegerType(), True),
        StructField("ingestion_timestamp", TimestampType(), False),
    ]
)

HEADERS = {"User-Agent": settings.REDDIT_USER_AGENT}


def fetch_subreddit_posts(subreddit: str, sort: str = "hot", limit: int = settings.POSTS_PER_SUBREDDIT) -> list[dict]:
    posts = []
    after = None
    now = datetime.now(timezone.utc)

    while len(posts) < limit:
        batch_size = min(100, limit - len(posts))
        url = settings.REDDIT_BASE_URL.format(subreddit=subreddit, sort=sort)
        params = {"limit": batch_size, "raw_json": 1}
        if after:
            params["after"] = after

        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            p = child["data"]
            posts.append(
                {
                    "post_id": p["id"],
                    "subreddit": subreddit,
                    "title": p["title"],
                    "selftext": p.get("selftext") or None,
                    "author": p.get("author", "[deleted]"),
                    "score": int(p.get("score", 0)),
                    "upvote_ratio": float(p.get("upvote_ratio", 0.0)),
                    "num_comments": int(p.get("num_comments", 0)),
                    "created_utc": datetime.fromtimestamp(p["created_utc"], tz=timezone.utc),
                    "url": p.get("url"),
                    "permalink": f"https://reddit.com{p['permalink']}",
                    "is_self": str(p.get("is_self", False)),
                    "link_flair_text": p.get("link_flair_text"),
                    "over_18": str(p.get("over_18", False)),
                    "spoiler": str(p.get("spoiler", False)),
                    "awards_received": int(p.get("total_awards_received", 0)),
                    "ingestion_timestamp": now,
                }
            )

        after = data["data"].get("after")
        if not after:
            break

        time.sleep(1.5)

    return posts


def fetch_all_subreddits() -> list[dict]:
    all_posts = []
    for sub in settings.SUBREDDITS:
        print(f"Fetching r/{sub}...")
        try:
            posts = fetch_subreddit_posts(sub)
            all_posts.extend(posts)
            print(f"  -> {len(posts)} posts")
        except Exception as e:
            print(f"  -> ERROR: {e}")
        time.sleep(2)
    print(f"\nTotal posts fetched: {len(all_posts)}")
    return all_posts


def ingest_to_bronze(spark: SparkSession, posts: list[dict]):
    df = spark.createDataFrame(posts, schema=POST_SCHEMA)
    df.write.format("delta").mode("append").option("mergeSchema", "true").save(settings.BRONZE_PATH)
    print(f"Wrote {df.count()} rows to {settings.BRONZE_PATH}")
    return df
