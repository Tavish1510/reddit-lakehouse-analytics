# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Vector Search Index
# MAGIC Creates a Databricks Vector Search endpoint and index over the
# MAGIC post embeddings table, enabling semantic search for the Streamlit Q&A app.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

CATALOG = "reddit_analytics"
EMBEDDING_TABLE = f"{CATALOG}.gold.post_embeddings"
VS_ENDPOINT = "reddit_vs_endpoint"
VS_INDEX = f"{CATALOG}.gold.post_embeddings_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Endpoint

# COMMAND ----------

vsc = VectorSearchClient()

try:
    vsc.get_endpoint(VS_ENDPOINT)
    print(f"Endpoint '{VS_ENDPOINT}' already exists")
except Exception:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print(f"Creating endpoint '{VS_ENDPOINT}'... (this may take a few minutes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Endpoint to Be Ready

# COMMAND ----------

import time

endpoint = vsc.get_endpoint(VS_ENDPOINT)
while endpoint.get("endpoint_status", {}).get("state") != "ONLINE":
    print(f"Endpoint status: {endpoint.get('endpoint_status', {}).get('state', 'UNKNOWN')}")
    time.sleep(30)
    endpoint = vsc.get_endpoint(VS_ENDPOINT)

print("Endpoint is ONLINE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Delta Sync Index
# MAGIC Syncs automatically when the source Delta table is updated.

# COMMAND ----------

try:
    vsc.get_index(VS_ENDPOINT, VS_INDEX)
    print(f"Index '{VS_INDEX}' already exists")
except Exception:
    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        index_name=VS_INDEX,
        source_table_name=EMBEDDING_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="post_id",
        embedding_dimension=1536,
        embedding_vector_column="embedding",
    )
    print(f"Creating index '{VS_INDEX}'...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Index Sync

# COMMAND ----------

index = vsc.get_index(VS_ENDPOINT, VS_INDEX)
while not index.describe().get("status", {}).get("ready"):
    print("Index syncing...")
    time.sleep(30)
    index = vsc.get_index(VS_ENDPOINT, VS_INDEX)

print("Index is ready for queries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Similarity Search

# COMMAND ----------

from openai import OpenAI

OPENAI_API_KEY = dbutils.secrets.get("reddit-api", "openai-key")
client = OpenAI(api_key=OPENAI_API_KEY)

test_query = "What are the best programming languages to learn in 2025?"

query_embedding = client.embeddings.create(
    model="text-embedding-3-small", input=[test_query]
).data[0].embedding

results = index.similarity_search(
    query_vector=query_embedding,
    columns=["post_id", "subreddit", "title", "content"],
    num_results=5,
)

print(f"Query: {test_query}\n")
for row in results.get("result", {}).get("data_array", []):
    print(f"  [{row[1]}] {row[2]}")
    print(f"  Score: {row[-1]:.4f}\n")
