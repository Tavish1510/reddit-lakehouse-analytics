# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Bronze Ingestion
# MAGIC Fetches posts from 16 subreddits via the Reddit API (PRAW)
# MAGIC and writes raw data into the Bronze Delta table.

# COMMAND ----------

# MAGIC %pip install praw python-dotenv
# MAGIC %restart_python

# COMMAND ----------

from datetime import datetime, timezone

import praw
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC Store your Reddit API credentials as Databricks secrets:
# MAGIC ```
# MAGIC databricks secrets create-scope reddit-api
# MAGIC databricks secrets put-secret reddit-api client-id --string-value "YOUR_CLIENT_ID"
# MAGIC databricks secrets put-secret reddit-api client-secret --string-value "YOUR_SECRET"
# MAGIC ```

# COMMAND ----------

REDDIT_CLIENT_ID = dbutils.secrets.get("reddit-api", "client-id")
REDDIT_CLIENT_SECRET = dbutils.secrets.get("reddit-api", "client-secret")
REDDIT_USER_AGENT = "RedditLakehouse/1.0 (Databricks)"

CATALOG = "reddit_analytics"
BRONZE_TABLE = f"{CATALOG}.bronze.raw_posts"

SUBREDDITS = [
    "technology", "programming", "MachineLearning", "datascience",
    "wallstreetbets", "personalfinance", "investing",
    "science", "space",
    "worldnews", "news",
    "AskReddit", "explainlikeimfive",
    "fitness", "cooking", "travel",
]

POSTS_PER_SUBREDDIT = 500

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Posts from Reddit

# COMMAND ----------

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

all_posts = []
now = datetime.now(timezone.utc)

for subreddit_name in SUBREDDITS:
    print(f"Fetching r/{subreddit_name}...")
    subreddit = reddit.subreddit(subreddit_name)

    for submission in subreddit.hot(limit=POSTS_PER_SUBREDDIT):
        all_posts.append({
            "post_id": submission.id,
            "subreddit": subreddit_name,
            "title": submission.title,
            "selftext": submission.selftext or None,
            "author": str(submission.author) if submission.author else "[deleted]",
            "score": int(submission.score),
            "upvote_ratio": float(submission.upvote_ratio),
            "num_comments": int(submission.num_comments),
            "created_utc": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
            "url": submission.url,
            "permalink": f"https://reddit.com{submission.permalink}",
            "is_self": str(submission.is_self),
            "link_flair_text": submission.link_flair_text,
            "over_18": str(submission.over_18),
            "spoiler": str(submission.spoiler),
            "awards_received": int(submission.total_awards_received),
            "ingestion_timestamp": now,
        })

    print(f"  -> {len([p for p in all_posts if p['subreddit'] == subreddit_name])} posts")

print(f"\nTotal posts fetched: {len(all_posts)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Bronze Delta Table

# COMMAND ----------

schema = StructType([
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
])

spark.sql(f"USE CATALOG {CATALOG}")

bronze_df = spark.createDataFrame(all_posts, schema=schema)
bronze_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("bronze.raw_posts")

print(f"Wrote {bronze_df.count()} rows to {BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Bronze Data

# COMMAND ----------

display(spark.sql(f"""
    SELECT subreddit, COUNT(*) as post_count
    FROM {BRONZE_TABLE}
    GROUP BY subreddit
    ORDER BY post_count DESC
"""))
