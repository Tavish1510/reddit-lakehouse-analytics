from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def build_subreddit_daily_stats(silver_df: DataFrame) -> DataFrame:
    return silver_df.groupBy("subreddit", "created_date").agg(
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


def build_subreddit_summary(silver_df: DataFrame) -> DataFrame:
    return silver_df.groupBy("subreddit").agg(
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


def build_hourly_activity(silver_df: DataFrame) -> DataFrame:
    return silver_df.groupBy("subreddit", "created_hour", "day_of_week").agg(
        F.count("post_id").alias("post_count"),
        F.avg("score").alias("avg_score"),
        F.avg("num_comments").alias("avg_comments"),
    )


def build_top_posts(silver_df: DataFrame, top_n: int = 50) -> DataFrame:
    from pyspark.sql.window import Window

    window = Window.partitionBy("subreddit").orderBy(F.col("score").desc())
    return (
        silver_df.withColumn("rank", F.row_number().over(window))
        .filter(F.col("rank") <= top_n)
        .select("subreddit", "rank", "post_id", "title", "author", "score", "num_comments", "upvote_ratio", "created_utc", "permalink", "content")
    )


def build_author_activity(silver_df: DataFrame) -> DataFrame:
    return (
        silver_df.filter(F.col("author").isNotNull())
        .groupBy("subreddit", "author")
        .agg(
            F.count("post_id").alias("post_count"),
            F.avg("score").alias("avg_score"),
            F.sum("score").alias("total_score"),
            F.avg("num_comments").alias("avg_comments"),
        )
    )


def transform_silver_to_gold(spark: SparkSession, silver_path: str, gold_base_path: str):
    silver_df = spark.read.format("delta").load(silver_path)
    silver_df.cache()

    tables = {
        "subreddit_daily_stats": build_subreddit_daily_stats(silver_df),
        "subreddit_summary": build_subreddit_summary(silver_df),
        "hourly_activity": build_hourly_activity(silver_df),
        "top_posts": build_top_posts(silver_df),
        "author_activity": build_author_activity(silver_df),
    }

    for name, df in tables.items():
        path = f"{gold_base_path}/{name}"
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)
        print(f"Gold table '{name}' written to {path} ({df.count()} rows)")

    silver_df.unpersist()
    return tables
