# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Gold Aggregations
# MAGIC Builds business-level aggregate tables from the Silver layer:
# MAGIC - Subreddit daily stats
# MAGIC - Subreddit summary
# MAGIC - Hourly activity heatmap
# MAGIC - Top posts per subreddit
# MAGIC - Author activity

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "reddit_analytics"
SILVER_TABLE = f"{CATALOG}.silver.cleaned_posts"

spark.sql(f"USE CATALOG {CATALOG}")
silver_df = spark.read.table("silver.cleaned_posts")
silver_df.cache()
print(f"Silver rows loaded: {silver_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Subreddit Daily Stats

# COMMAND ----------

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

daily_stats.write.format("delta").mode("overwrite").saveAsTable("gold.subreddit_daily_stats")
print(f"Daily stats: {daily_stats.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Subreddit Summary

# COMMAND ----------

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

subreddit_summary.write.format("delta").mode("overwrite").saveAsTable("gold.subreddit_summary")
display(subreddit_summary.orderBy(F.col("total_posts").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hourly Activity (for Heatmap Visualizations)

# COMMAND ----------

hourly_activity = silver_df.groupBy("subreddit", "created_hour", "day_of_week").agg(
    F.count("post_id").alias("post_count"),
    F.avg("score").alias("avg_score"),
    F.avg("num_comments").alias("avg_comments"),
)

hourly_activity.write.format("delta").mode("overwrite").saveAsTable("gold.hourly_activity")
print(f"Hourly activity: {hourly_activity.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top 50 Posts Per Subreddit

# COMMAND ----------

window = Window.partitionBy("subreddit").orderBy(F.col("score").desc())

top_posts = (
    silver_df
    .withColumn("rank", F.row_number().over(window))
    .filter(F.col("rank") <= 50)
    .select(
        "subreddit", "rank", "post_id", "title", "author",
        "score", "num_comments", "upvote_ratio",
        "created_utc", "permalink", "content"
    )
)

top_posts.write.format("delta").mode("overwrite").saveAsTable("gold.top_posts")
print(f"Top posts: {top_posts.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Author Activity

# COMMAND ----------

author_activity = (
    silver_df
    .filter(F.col("author").isNotNull())
    .groupBy("subreddit", "author")
    .agg(
        F.count("post_id").alias("post_count"),
        F.avg("score").alias("avg_score"),
        F.sum("score").alias("total_score"),
        F.avg("num_comments").alias("avg_comments"),
    )
)

author_activity.write.format("delta").mode("overwrite").saveAsTable("gold.author_activity")
print(f"Author activity: {author_activity.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Gold Tables

# COMMAND ----------

gold_tables = ["subreddit_daily_stats", "subreddit_summary", "hourly_activity", "top_posts", "author_activity"]

for table in gold_tables:
    spark.sql(f"OPTIMIZE gold.{table}")
    print(f"Optimized gold.{table}")

# COMMAND ----------

silver_df.unpersist()
print("Gold layer build complete.")
