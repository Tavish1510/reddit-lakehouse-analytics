"""
02 — Bronze Ingestion
Fetches posts from 16 subreddits via Reddit's public JSON API
and writes raw data into the Bronze Delta table.
No API key required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from pyspark.sql import SparkSession

from src.ingestion.reddit_client import POST_SCHEMA, fetch_all_subreddits

# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder.appName("RedditLakehouse-Bronze")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .master("local[*]")
    .getOrCreate()
)

# ---------------------------------------------------------------------------
# Fetch posts from Reddit (public JSON, no API key)
# ---------------------------------------------------------------------------
print("=" * 60)
print("BRONZE INGESTION — Reddit Public JSON API")
print("=" * 60)

all_posts = fetch_all_subreddits()

# ---------------------------------------------------------------------------
# Write to Bronze Delta Table
# ---------------------------------------------------------------------------
Path(settings.BRONZE_PATH).mkdir(parents=True, exist_ok=True)

bronze_df = spark.createDataFrame(all_posts, schema=POST_SCHEMA)
bronze_df.write.format("delta").mode("append").option("mergeSchema", "true").save(settings.BRONZE_PATH)

print(f"\nWrote {bronze_df.count()} rows to Bronze: {settings.BRONZE_PATH}")

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
print("\nPosts per subreddit:")
bronze_df.groupBy("subreddit").count().orderBy("count", ascending=False).show(20, truncate=False)

spark.stop()
