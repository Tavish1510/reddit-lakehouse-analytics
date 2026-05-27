"""
01 — Environment Setup
Creates the local Delta Lake directory structure (Bronze / Silver / Gold)
and initializes a SparkSession with Delta Lake support.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from pyspark.sql import SparkSession

# ---------------------------------------------------------------------------
# Create directory structure
# ---------------------------------------------------------------------------
for path in [settings.BRONZE_PATH, settings.SILVER_PATH, settings.GOLD_DIR, settings.CHROMA_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {path}")

# ---------------------------------------------------------------------------
# Verify SparkSession with Delta Lake
# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder.appName("RedditLakehouse")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .master("local[*]")
    .getOrCreate()
)

print(f"\nSpark version: {spark.version}")
print(f"Delta Lake enabled: {spark.conf.get('spark.sql.extensions')}")
print("\nEnvironment setup complete.")

spark.stop()
