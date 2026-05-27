"""
05 — Embedding Generation
Generates embeddings for Reddit post content using the free
HuggingFace Sentence-Transformers model (all-MiniLM-L6-v2)
and stores them in a ChromaDB vector database.
No API keys required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.embeddings.generator import build_vector_store

# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder.appName("RedditLakehouse-Embeddings")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .master("local[*]")
    .getOrCreate()
)

# ---------------------------------------------------------------------------
# Select top posts per subreddit for embedding
# ---------------------------------------------------------------------------
print("=" * 60)
print("EMBEDDING GENERATION — Sentence-Transformers (free, local)")
print("=" * 60)

silver_df = spark.read.format("delta").load(settings.SILVER_PATH)

window = Window.partitionBy("subreddit").orderBy(F.col("score").desc())
posts_to_embed = (
    silver_df.withColumn("rank", F.row_number().over(window))
    .filter(F.col("rank") <= 100)
    .select("post_id", "subreddit", "title", "content")
)

print(f"Posts to embed: {posts_to_embed.count()}")

# ---------------------------------------------------------------------------
# Build ChromaDB vector store
# ---------------------------------------------------------------------------
Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
collection = build_vector_store(posts_to_embed)

# ---------------------------------------------------------------------------
# Test query
# ---------------------------------------------------------------------------
print("\n--- Test Query ---")
from src.embeddings.generator import search

results = search("What are the best programming languages to learn?", n_results=5)
for i, r in enumerate(results, 1):
    print(f"  {i}. [r/{r['subreddit']}] {r['title']}")
    print(f"     distance: {r['distance']:.4f}")

spark.stop()
