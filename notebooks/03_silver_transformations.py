# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Silver Transformations
# MAGIC Reads Bronze raw posts, applies cleaning, deduplication, normalization,
# MAGIC and writes enriched data into the Silver Delta table.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "reddit_analytics"
BRONZE_TABLE = f"{CATALOG}.bronze.raw_posts"
SILVER_TABLE = f"{CATALOG}.silver.cleaned_posts"

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Bronze

# COMMAND ----------

bronze_df = spark.read.table("bronze.raw_posts")
print(f"Bronze rows: {bronze_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Deduplicate
# MAGIC Keep only the latest ingestion of each post.

# COMMAND ----------

window = Window.partitionBy("post_id").orderBy(F.col("ingestion_timestamp").desc())
deduped_df = (
    bronze_df
    .withColumn("row_num", F.row_number().over(window))
    .filter(F.col("row_num") == 1)
    .drop("row_num")
)

removed = bronze_df.count() - deduped_df.count()
print(f"Removed {removed} duplicate rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Clean Text Fields

# COMMAND ----------

cleaned_df = (
    deduped_df
    .withColumn("title", F.trim(F.regexp_replace("title", r"[\x00-\x1f\x7f]", "")))
    .withColumn(
        "selftext",
        F.when(F.col("selftext").isNotNull(),
               F.trim(F.regexp_replace("selftext", r"[\x00-\x1f\x7f]", "")))
        .otherwise(F.lit(None))
    )
    .withColumn(
        "selftext",
        F.when(
            (F.col("selftext") == "") |
            (F.col("selftext") == "[removed]") |
            (F.col("selftext") == "[deleted]"),
            F.lit(None)
        ).otherwise(F.col("selftext"))
    )
    .withColumn(
        "author",
        F.when(F.col("author") == "[deleted]", F.lit(None)).otherwise(F.col("author"))
    )
    .withColumn(
        "link_flair_text",
        F.when(F.col("link_flair_text").isNotNull(),
               F.lower(F.trim("link_flair_text")))
        .otherwise(F.lit("unflaired"))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Normalize Types

# COMMAND ----------

normalized_df = (
    cleaned_df
    .withColumn("is_self", F.col("is_self").cast("boolean"))
    .withColumn("over_18", F.col("over_18").cast("boolean"))
    .withColumn("spoiler", F.col("spoiler").cast("boolean"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Add Derived Columns

# COMMAND ----------

silver_df = (
    normalized_df
    .withColumn("title_length", F.length("title"))
    .withColumn(
        "selftext_length",
        F.when(F.col("selftext").isNotNull(), F.length("selftext")).otherwise(F.lit(0))
    )
    .withColumn("has_selftext", F.col("selftext").isNotNull())
    .withColumn(
        "content",
        F.when(F.col("selftext").isNotNull(),
               F.concat_ws(" ", F.col("title"), F.col("selftext")))
        .otherwise(F.col("title"))
    )
    .withColumn("created_date", F.to_date("created_utc"))
    .withColumn("created_hour", F.hour("created_utc"))
    .withColumn("day_of_week", F.dayofweek("created_utc"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Silver (Partitioned by Subreddit)

# COMMAND ----------

silver_df.write.format("delta").mode("overwrite").partitionBy("subreddit").option("overwriteSchema", "true").saveAsTable("silver.cleaned_posts")

print(f"Silver table written: {silver_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Silver Table

# COMMAND ----------

spark.sql(f"OPTIMIZE {SILVER_TABLE} ZORDER BY (created_date, score)")
print("Z-ORDER optimization complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Silver

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        subreddit,
        COUNT(*) as post_count,
        ROUND(AVG(score), 1) as avg_score,
        ROUND(AVG(title_length), 0) as avg_title_len,
        SUM(CASE WHEN has_selftext THEN 1 ELSE 0 END) as text_posts,
        SUM(CASE WHEN author IS NULL THEN 1 ELSE 0 END) as deleted_authors
    FROM {SILVER_TABLE}
    GROUP BY subreddit
    ORDER BY post_count DESC
"""))
