"""
Shared Spark session helper with Delta Lake configured.

Includes Windows-specific networking fixes (driver bind address, host)
so the pipeline runs reliably on Windows, Mac, and Linux.
"""

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "RedditLakehouse") -> SparkSession:
    venv_python = Path(sys.executable)
    os.environ["PYSPARK_PYTHON"] = str(venv_python)
    os.environ["PYSPARK_DRIVER_PYTHON"] = str(venv_python)

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.python.worker.reuse", "true")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.sql.shuffle.partitions", "8")
        .master("local[*]")
    )

    return builder.getOrCreate()
