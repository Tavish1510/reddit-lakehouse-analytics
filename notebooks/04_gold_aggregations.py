"""
04 — Gold Aggregations
Builds business-level aggregate tables from the Silver layer:
  - Subreddit daily stats
  - Subreddit summary
  - Hourly activity heatmap
  - Top posts per subreddit
  - Author activity
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder.appName("RedditLakehouse-Gold")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .master("local[*]")
    .getOrCreate()
)

# ---------------------------------------------------------------------------
# Read Silver
# ---------------------------------------------------------------------------
print("=" * 60)
print("GOLD AGGREGATIONS")
print("=" * 60)

silver_df = spark.read.format("delta").load(settings.SILVER_PATH)
silver_df.cache()
print(f"Silver rows loaded: {silver_df.count()}")

gold_dir = Path(settings.GOLD_DIR)
gold_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Subreddit Daily Stats
# ---------------------------------------------------------------------------
daily_stats = silver_df.groupBy("subreddit", "created_date").agg(
    F.count("post_id").alias("post_count"),
    F.avg("score").alias("avg_score"),
    F.avg("num_comments").alias("avg_comments"),
    F.avg("upvote_ratio").alias("avg_upvote_ratio"),
    F.sum("score").alias("total_score"),
    F.sum("num_comments").alias("total_comments"),
    F.avg("title_length").alias("avg_title_length"),
    F.avg("selftext_length").alias("avg_selftext_length"),
    F.sum(F.when(F.col("has_selftext"), 1).otherwise(0)).alias("text_posts"),
    F.sum(F.when(~F.col("has_selftext"), 1).otherwise(0)).alias("link_posts"),
)

daily_path = str(gold_dir / "subreddit_daily_stats")
daily_stats.write.format("delta").mode("overwrite").save(daily_path)
print(f"1. subreddit_daily_stats: {daily_stats.count()} rows")

# ---------------------------------------------------------------------------
# 2. Subreddit Summary
# ---------------------------------------------------------------------------
subreddit_summary = silver_df.groupBy("subreddit").agg(
    F.count("post_id").alias("total_posts"),
    F.avg("score").alias("avg_score"),
    F.stddev("score").alias("stddev_score"),
    F.max("score").alias("max_score"),
    F.avg("num_comments").alias("avg_comments"),
    F.max("num_comments").alias("max_comments"),
    F.avg("upvote_ratio").alias("avg_upvote_ratio"),
    F.countDistinct("author").alias("unique_authors"),
    F.avg("title_length").alias("avg_title_length"),
    F.sum(F.when(F.col("over_18"), 1).otherwise(0)).alias("nsfw_count"),
)

summary_path = str(gold_dir / "subreddit_summary")
subreddit_summary.write.format("delta").mode("overwrite").save(summary_path)
print(f"2. subreddit_summary: {subreddit_summary.count()} rows")
subreddit_summary.orderBy(F.col("total_posts").desc()).show(20, truncate=False)

# ---------------------------------------------------------------------------
# 3. Hourly Activity
# ---------------------------------------------------------------------------
hourly_activity = silver_df.groupBy("subreddit", "created_hour", "day_of_week").agg(
    F.count("post_id").alias("post_count"),
    F.avg("score").alias("avg_score"),
    F.avg("num_comments").alias("avg_comments"),
)

hourly_path = str(gold_dir / "hourly_activity")
hourly_activity.write.format("delta").mode("overwrite").save(hourly_path)
print(f"3. hourly_activity: {hourly_activity.count()} rows")

# ---------------------------------------------------------------------------
# 4. Top 50 Posts Per Subreddit
# ---------------------------------------------------------------------------
window = Window.partitionBy("subreddit").orderBy(F.col("score").desc())
top_posts = (
    silver_df.withColumn("rank", F.row_number().over(window))
    .filter(F.col("rank") <= 50)
    .select("subreddit", "rank", "post_id", "title", "author", "score", "num_comments", "upvote_ratio", "created_utc", "permalink", "content")
)

top_path = str(gold_dir / "top_posts")
top_posts.write.format("delta").mode("overwrite").save(top_path)
print(f"4. top_posts: {top_posts.count()} rows")

# ---------------------------------------------------------------------------
# 5. Author Activity
# ---------------------------------------------------------------------------
author_activity = (
    silver_df.filter(F.col("author").isNotNull())
    .groupBy("subreddit", "author")
    .agg(
        F.count("post_id").alias("post_count"),
        F.avg("score").alias("avg_score"),
        F.sum("score").alias("total_score"),
        F.avg("num_comments").alias("avg_comments"),
    )
)

author_path = str(gold_dir / "author_activity")
author_activity.write.format("delta").mode("overwrite").save(author_path)
print(f"5. author_activity: {author_activity.count()} rows")

# ---------------------------------------------------------------------------
silver_df.unpersist()
print("\nGold layer build complete.")
spark.stop()
