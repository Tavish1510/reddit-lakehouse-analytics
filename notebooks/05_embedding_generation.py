# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Embedding Generation
# MAGIC Generates OpenAI embeddings for Reddit post content (title + selftext)
# MAGIC and stores them in the Gold layer for vector search.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC ```
# MAGIC databricks secrets put-secret reddit-api openai-key --string-value "YOUR_OPENAI_KEY"
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install openai
# MAGIC %restart_python

# COMMAND ----------

import time

from openai import OpenAI
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

# COMMAND ----------

OPENAI_API_KEY = dbutils.secrets.get("reddit-api", "openai-key")
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

CATALOG = "reddit_analytics"
SILVER_TABLE = f"{CATALOG}.silver.cleaned_posts"
EMBEDDING_TABLE = f"{CATALOG}.gold.post_embeddings"

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select Posts for Embedding
# MAGIC Use the top posts from each subreddit to keep API costs manageable.

# COMMAND ----------

from pyspark.sql.window import Window

window = Window.partitionBy("subreddit").orderBy(F.col("score").desc())

posts_to_embed = (
    spark.read.table("silver.cleaned_posts")
    .withColumn("rank", F.row_number().over(window))
    .filter(F.col("rank") <= 100)
    .select("post_id", "subreddit", "title", "content")
    .drop("rank")
)

post_count = posts_to_embed.count()
print(f"Posts to embed: {post_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Embeddings

# COMMAND ----------

client = OpenAI(api_key=OPENAI_API_KEY)
rows = posts_to_embed.collect()

embedded_rows = []
for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i:i + BATCH_SIZE]
    texts = [row["content"][:8000] for row in batch]

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embeddings = [item.embedding for item in response.data]

    for row, embedding in zip(batch, embeddings):
        embedded_rows.append({
            "post_id": row["post_id"],
            "subreddit": row["subreddit"],
            "title": row["title"],
            "content": row["content"],
            "embedding": embedding,
        })

    print(f"Embedded {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")
    if i + BATCH_SIZE < len(rows):
        time.sleep(0.5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Embeddings to Gold

# COMMAND ----------

schema = StructType([
    StructField("post_id", StringType(), False),
    StructField("subreddit", StringType(), False),
    StructField("title", StringType(), False),
    StructField("content", StringType(), False),
    StructField("embedding", ArrayType(FloatType()), False),
])

embeddings_df = spark.createDataFrame(embedded_rows, schema=schema)
embeddings_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("gold.post_embeddings")

print(f"Wrote {embeddings_df.count()} embeddings to {EMBEDDING_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate

# COMMAND ----------

display(spark.sql(f"""
    SELECT subreddit, COUNT(*) as embedded_posts,
           SIZE(embedding) as embedding_dim
    FROM {EMBEDDING_TABLE}
    GROUP BY subreddit
    ORDER BY embedded_posts DESC
"""))
