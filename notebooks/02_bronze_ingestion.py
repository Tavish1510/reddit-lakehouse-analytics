"""
02 — Bronze Ingestion
Fetches posts from 16 subreddits via Reddit's public JSON API,
writes them to a staging JSON file, then loads into the Bronze
Delta table via Spark.
No API key required.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType

from src.ingestion.reddit_client import POST_SCHEMA, fetch_all_subreddits
from src.spark_session import get_spark_session

# ---------------------------------------------------------------------------
# Fetch posts from Reddit (public JSON, no API key)
# ---------------------------------------------------------------------------
print("=" * 60)
print("BRONZE INGESTION — Reddit Public JSON API")
print("=" * 60)

all_posts = fetch_all_subreddits()
print(f"\nFetched {len(all_posts)} posts total")

if not all_posts:
    print("\nNo posts fetched — exiting. (Reddit may be rate-limiting — wait 5-10 min and retry.)")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Stage to JSON (avoids Python-worker bottleneck when calling
# spark.createDataFrame on a large Python list on Windows)
# ---------------------------------------------------------------------------
staging_path = Path(settings.BRONZE_PATH).parent / "_staging" / "posts.json"
staging_path.parent.mkdir(parents=True, exist_ok=True)


def serialize(post: dict) -> dict:
    out = dict(post)
    for key in ("created_utc", "ingestion_timestamp"):
        if isinstance(out[key], datetime):
            out[key] = out[key].isoformat()
    return out


with staging_path.open("w", encoding="utf-8") as f:
    for post in all_posts:
        f.write(json.dumps(serialize(post)) + "\n")

print(f"Staged to: {staging_path}")

# ---------------------------------------------------------------------------
# Load JSON into Spark and write to Bronze Delta
# ---------------------------------------------------------------------------
spark = get_spark_session("RedditLakehouse-Bronze")

raw_df = spark.read.json(str(staging_path))

bronze_df = (
    raw_df
    .withColumn("created_utc", F.col("created_utc").cast(TimestampType()))
    .withColumn("ingestion_timestamp", F.col("ingestion_timestamp").cast(TimestampType()))
    .select(*[field.name for field in POST_SCHEMA.fields])
)

Path(settings.BRONZE_PATH).mkdir(parents=True, exist_ok=True)
bronze_df.write.format("delta").mode("append").option("mergeSchema", "true").save(settings.BRONZE_PATH)

print(f"\nWrote {bronze_df.count()} rows to Bronze: {settings.BRONZE_PATH}")

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
print("\nPosts per subreddit:")
bronze_df.groupBy("subreddit").count().orderBy("count", ascending=False).show(20, truncate=False)

spark.stop()
