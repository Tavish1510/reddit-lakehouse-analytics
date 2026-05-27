from datetime import datetime, timezone

import praw
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from config.settings import Settings

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


class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=Settings.REDDIT_CLIENT_ID,
            client_secret=Settings.REDDIT_CLIENT_SECRET,
            user_agent=Settings.REDDIT_USER_AGENT,
        )

    def fetch_posts(self, subreddit_name: str, limit: int = Settings.POSTS_PER_SUBREDDIT) -> list[dict]:
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []
        now = datetime.now(timezone.utc)

        for submission in subreddit.hot(limit=limit):
            posts.append(
                {
                    "post_id": submission.id,
                    "subreddit": subreddit_name,
                    "title": submission.title,
                    "selftext": submission.selftext or None,
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "score": submission.score,
                    "upvote_ratio": submission.upvote_ratio,
                    "num_comments": submission.num_comments,
                    "created_utc": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                    "url": submission.url,
                    "permalink": f"https://reddit.com{submission.permalink}",
                    "is_self": str(submission.is_self),
                    "link_flair_text": submission.link_flair_text,
                    "over_18": str(submission.over_18),
                    "spoiler": str(submission.spoiler),
                    "awards_received": submission.total_awards_received,
                    "ingestion_timestamp": now,
                }
            )

        return posts

    def fetch_all_subreddits(self) -> list[dict]:
        all_posts = []
        for sub in Settings.SUBREDDITS:
            print(f"Fetching posts from r/{sub}...")
            posts = self.fetch_posts(sub)
            all_posts.extend(posts)
            print(f"  -> {len(posts)} posts fetched")
        print(f"Total posts fetched: {len(all_posts)}")
        return all_posts


def ingest_to_bronze(spark: SparkSession, posts: list[dict], table_path: str):
    df = spark.createDataFrame(posts, schema=POST_SCHEMA)
    df.write.format("delta").mode("append").option("mergeSchema", "true").save(table_path)
    return df
