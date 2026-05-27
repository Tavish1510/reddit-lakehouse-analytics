"""
03 — Silver Transformations
Reads Bronze raw posts, applies cleaning, deduplication, normalization,
and writes enriched data into the Silver Delta table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.spark_session import get_spark_session

# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------
spark = get_spark_session("RedditLakehouse-Silver")

# ---------------------------------------------------------------------------
# Read Bronze
# ---------------------------------------------------------------------------
print("=" * 60)
print("SILVER TRANSFORMATIONS")
print("=" * 60)

bronze_df = spark.read.format("delta").load(settings.BRONZE_PATH)
print(f"Bronze rows: {bronze_df.count()}")

# ---------------------------------------------------------------------------
# Step 1: Deduplicate (keep latest ingestion per post)
# ---------------------------------------------------------------------------
window = Window.partitionBy("post_id").orderBy(F.col("ingestion_timestamp").desc())
deduped_df = bronze_df.withColumn("row_num", F.row_number().over(window)).filter(F.col("row_num") == 1).drop("row_num")

removed = bronze_df.count() - deduped_df.count()
print(f"Step 1 — Deduplication: removed {removed} duplicate rows")

# ---------------------------------------------------------------------------
# Step 2: Clean text fields
# ---------------------------------------------------------------------------
cleaned_df = (
    deduped_df
    .withColumn("title", F.trim(F.regexp_replace("title", r"[\x00-\x1f\x7f]", "")))
    .withColumn(
        "selftext",
        F.when(F.col("selftext").isNotNull(), F.trim(F.regexp_replace("selftext", r"[\x00-\x1f\x7f]", "")))
        .otherwise(F.lit(None)),
    )
    .withColumn(
        "selftext",
        F.when(
            (F.col("selftext") == "") | (F.col("selftext") == "[removed]") | (F.col("selftext") == "[deleted]"),
            F.lit(None),
        ).otherwise(F.col("selftext")),
    )
    .withColumn("author", F.when(F.col("author") == "[deleted]", F.lit(None)).otherwise(F.col("author")))
    .withColumn(
        "link_flair_text",
        F.when(F.col("link_flair_text").isNotNull(), F.lower(F.trim("link_flair_text"))).otherwise(F.lit("unflaired")),
    )
)

print("Step 2 — Text cleaning: done")

# ---------------------------------------------------------------------------
# Step 3: Normalize types
# ---------------------------------------------------------------------------
normalized_df = (
    cleaned_df
    .withColumn("is_self", F.col("is_self").cast("boolean"))
    .withColumn("over_18", F.col("over_18").cast("boolean"))
    .withColumn("spoiler", F.col("spoiler").cast("boolean"))
)

print("Step 3 — Type normalization: done")

# ---------------------------------------------------------------------------
# Step 4: Add derived columns
# ---------------------------------------------------------------------------
silver_df = (
    normalized_df
    .withColumn("title_length", F.length("title"))
    .withColumn("selftext_length", F.when(F.col("selftext").isNotNull(), F.length("selftext")).otherwise(F.lit(0)))
    .withColumn("has_selftext", F.col("selftext").isNotNull())
    .withColumn(
        "content",
        F.when(F.col("selftext").isNotNull(), F.concat_ws(" ", F.col("title"), F.col("selftext"))).otherwise(F.col("title")),
    )
    .withColumn("created_date", F.to_date("created_utc"))
    .withColumn("created_hour", F.hour("created_utc"))
    .withColumn("day_of_week", F.dayofweek("created_utc"))
)

print("Step 4 — Derived columns: done")

# ---------------------------------------------------------------------------
# Write to Silver (partitioned by subreddit)
# ---------------------------------------------------------------------------
Path(settings.SILVER_PATH).mkdir(parents=True, exist_ok=True)
silver_df.write.format("delta").mode("overwrite").partitionBy("subreddit").option("overwriteSchema", "true").save(settings.SILVER_PATH)

print(f"\nSilver table written: {silver_df.count()} rows -> {settings.SILVER_PATH}")

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
print("\nSilver summary:")
silver_df.groupBy("subreddit").agg(
    F.count("post_id").alias("posts"),
    F.round(F.avg("score"), 1).alias("avg_score"),
    F.round(F.avg("title_length"), 0).alias("avg_title_len"),
    F.sum(F.when(F.col("has_selftext"), 1).otherwise(0)).alias("text_posts"),
).orderBy("posts", ascending=False).show(20, truncate=False)

spark.stop()
