import time

from openai import OpenAI
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType

from config.settings import Settings


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=Settings.OPENAI_API_KEY)


def generate_embeddings_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    response = client.embeddings.create(
        model=Settings.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_dataframe(spark: SparkSession, df: DataFrame, text_column: str = "content", batch_size: int = 100) -> DataFrame:
    client = get_openai_client()
    rows = df.select("post_id", "subreddit", "title", text_column).collect()

    embedded_rows = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [row[text_column] for row in batch]
        embeddings = generate_embeddings_batch(texts, client)

        for row, embedding in zip(batch, embeddings):
            embedded_rows.append(
                {
                    "post_id": row["post_id"],
                    "subreddit": row["subreddit"],
                    "title": row["title"],
                    "content": row[text_column],
                    "embedding": embedding,
                }
            )

        print(f"Embedded {min(i + batch_size, len(rows))}/{len(rows)} posts")
        if i + batch_size < len(rows):
            time.sleep(0.5)

    from pyspark.sql.types import StringType, StructField, StructType

    schema = StructType(
        [
            StructField("post_id", StringType(), False),
            StructField("subreddit", StringType(), False),
            StructField("title", StringType(), False),
            StructField("content", StringType(), False),
            StructField("embedding", ArrayType(FloatType()), False),
        ]
    )

    return spark.createDataFrame(embedded_rows, schema=schema)
