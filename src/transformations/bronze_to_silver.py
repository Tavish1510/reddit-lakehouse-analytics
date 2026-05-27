from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def deduplicate_posts(df: DataFrame) -> DataFrame:
    window = Window.partitionBy("post_id").orderBy(F.col("ingestion_timestamp").desc())
    return df.withColumn("row_num", F.row_number().over(window)).filter(F.col("row_num") == 1).drop("row_num")


def clean_text_fields(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("title", F.trim(F.regexp_replace("title", r"[\x00-\x1f\x7f]", "")))
        .withColumn("selftext", F.when(F.col("selftext").isNotNull(), F.trim(F.regexp_replace("selftext", r"[\x00-\x1f\x7f]", ""))).otherwise(F.lit(None)))
        .withColumn("selftext", F.when((F.col("selftext") == "") | (F.col("selftext") == "[removed]") | (F.col("selftext") == "[deleted]"), F.lit(None)).otherwise(F.col("selftext")))
        .withColumn("author", F.when(F.col("author") == "[deleted]", F.lit(None)).otherwise(F.col("author")))
        .withColumn("link_flair_text", F.when(F.col("link_flair_text").isNotNull(), F.lower(F.trim("link_flair_text"))).otherwise(F.lit("unflaired")))
    )


def normalize_types(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("is_self", F.col("is_self").cast("boolean"))
        .withColumn("over_18", F.col("over_18").cast("boolean"))
        .withColumn("spoiler", F.col("spoiler").cast("boolean"))
    )


def add_derived_columns(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("title_length", F.length("title"))
        .withColumn("selftext_length", F.when(F.col("selftext").isNotNull(), F.length("selftext")).otherwise(F.lit(0)))
        .withColumn("has_selftext", F.col("selftext").isNotNull())
        .withColumn("content", F.when(F.col("selftext").isNotNull(), F.concat_ws(" ", F.col("title"), F.col("selftext"))).otherwise(F.col("title")))
        .withColumn("created_date", F.to_date("created_utc"))
        .withColumn("created_hour", F.hour("created_utc"))
        .withColumn("day_of_week", F.dayofweek("created_utc"))
    )


def transform_bronze_to_silver(spark: SparkSession, bronze_path: str, silver_path: str):
    bronze_df = spark.read.format("delta").load(bronze_path)

    silver_df = bronze_df.transform(deduplicate_posts).transform(clean_text_fields).transform(normalize_types).transform(add_derived_columns)

    silver_df.write.format("delta").mode("overwrite").partitionBy("subreddit").option("overwriteSchema", "true").save(silver_path)

    return silver_df
