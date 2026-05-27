# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Unity Catalog Setup
# MAGIC Creates the catalog, schemas (bronze/silver/gold), and registers Delta tables
# MAGIC for the Reddit Lakehouse Analytics pipeline.

# COMMAND ----------

CATALOG = "reddit_analytics"
SCHEMAS = ["bronze", "silver", "gold"]

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")

for schema in SCHEMAS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    print(f"Schema '{CATALOG}.{schema}' ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Table

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS bronze.raw_posts (
    post_id STRING NOT NULL,
    subreddit STRING NOT NULL,
    title STRING NOT NULL,
    selftext STRING,
    author STRING,
    score INT,
    upvote_ratio FLOAT,
    num_comments INT,
    created_utc TIMESTAMP NOT NULL,
    url STRING,
    permalink STRING,
    is_self STRING,
    link_flair_text STRING,
    over_18 STRING,
    spoiler STRING,
    awards_received INT,
    ingestion_timestamp TIMESTAMP NOT NULL
)
USING DELTA
COMMENT 'Raw Reddit posts ingested from PRAW API'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Table

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS silver.cleaned_posts (
    post_id STRING NOT NULL,
    subreddit STRING NOT NULL,
    title STRING NOT NULL,
    selftext STRING,
    author STRING,
    score INT,
    upvote_ratio FLOAT,
    num_comments INT,
    created_utc TIMESTAMP NOT NULL,
    url STRING,
    permalink STRING,
    is_self BOOLEAN,
    link_flair_text STRING,
    over_18 BOOLEAN,
    spoiler BOOLEAN,
    awards_received INT,
    ingestion_timestamp TIMESTAMP NOT NULL,
    title_length INT,
    selftext_length INT,
    has_selftext BOOLEAN,
    content STRING,
    created_date DATE,
    created_hour INT,
    day_of_week INT
)
USING DELTA
PARTITIONED BY (subreddit)
COMMENT 'Cleaned, deduplicated, and enriched Reddit posts'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Tables

# COMMAND ----------

spark.sql("""
CREATE TABLE IF NOT EXISTS gold.subreddit_daily_stats (
    subreddit STRING,
    created_date DATE,
    post_count LONG,
    avg_score DOUBLE,
    avg_comments DOUBLE,
    avg_upvote_ratio DOUBLE,
    total_score LONG,
    total_comments LONG,
    avg_title_length DOUBLE,
    avg_selftext_length DOUBLE,
    text_posts LONG,
    link_posts LONG
)
USING DELTA
COMMENT 'Daily aggregated stats per subreddit'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS gold.subreddit_summary (
    subreddit STRING,
    total_posts LONG,
    avg_score DOUBLE,
    stddev_score DOUBLE,
    max_score INT,
    avg_comments DOUBLE,
    max_comments INT,
    avg_upvote_ratio DOUBLE,
    unique_authors LONG,
    avg_title_length DOUBLE,
    nsfw_count LONG
)
USING DELTA
COMMENT 'Overall summary stats per subreddit'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS gold.top_posts (
    subreddit STRING,
    rank INT,
    post_id STRING,
    title STRING,
    author STRING,
    score INT,
    num_comments INT,
    upvote_ratio FLOAT,
    created_utc TIMESTAMP,
    permalink STRING,
    content STRING
)
USING DELTA
COMMENT 'Top 50 posts per subreddit by score'
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS gold.post_embeddings (
    post_id STRING,
    subreddit STRING,
    title STRING,
    content STRING,
    embedding ARRAY<FLOAT>
)
USING DELTA
COMMENT 'OpenAI embeddings for semantic search'
""")

# COMMAND ----------

print("All tables created successfully in Unity Catalog.")
spark.sql(f"SHOW TABLES IN {CATALOG}.bronze").show()
spark.sql(f"SHOW TABLES IN {CATALOG}.silver").show()
spark.sql(f"SHOW TABLES IN {CATALOG}.gold").show()
