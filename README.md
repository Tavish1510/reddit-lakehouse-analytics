# Reddit Lakehouse Analytics & Semantic Search Platform

https://reddit-lakehouse-analytics-kjcdrfnrdfvxkwe3tsqvu5.streamlit.app/

An end-to-end data pipeline using **PySpark** and **Delta Lake** that ingests Reddit posts from 16 subreddits, transforms them through a **Bronze/Silver/Gold medallion architecture**, and powers a **semantic search** system using Sentence-Transformers embeddings and ChromaDB.

**Fully free and local** — no cloud accounts, no API keys, no costs.

## Architecture

```
┌─────────────────┐     ┌─────────────────────────────────────────────┐     ┌──────────────┐
│  Reddit Public   │────▶│         Local Delta Lakehouse               │────▶│  Streamlit   │
│  JSON API        │     │                                             │     │  Search App  │
└─────────────────┘     │  ┌─────────┐  ┌──────────┐  ┌───────────┐  │     └──────┬───────┘
                        │  │ BRONZE  │─▶│  SILVER  │─▶│   GOLD    │  │            │
                        │  │ Raw     │  │ Cleaned  │  │ Aggregated│  │            │
                        │  │ Posts   │  │ Deduped  │  │ Analytics │  │            │
                        │  └─────────┘  └──────────┘  └───────────┘  │            │
                        │                                             │            │
                        └─────────────────────────────────────────────┘            │
                                                                                   │
                        ┌─────────────────────────────────────────────┐            │
                        │  Sentence-Transformers  ──▶  ChromaDB      │◀───────────┘
                        │  (all-MiniLM-L6-v2)        (Vector Store)  │
                        └─────────────────────────────────────────────┘
```

## Key Features

- **16 Subreddits** ingested: technology, programming, MachineLearning, datascience, wallstreetbets, personalfinance, investing, science, space, worldnews, news, AskReddit, explainlikeimfive, fitness, cooking, travel
- **Medallion Architecture**: Bronze (raw) → Silver (cleaned, deduplicated, normalized) → Gold (5 aggregated analytics tables)
- **PySpark ETL**: Scalable transformations with deduplication, text cleaning, type normalization, and derived feature engineering
- **Delta Lake**: ACID transactions, schema enforcement, partitioning by subreddit
- **Semantic Search**: HuggingFace `all-MiniLM-L6-v2` embeddings stored in ChromaDB
- **Streamlit App**: Interactive semantic search with subreddit filtering
- **Zero Cost**: No API keys, no cloud accounts — runs entirely on your machine

## Project Structure

```
reddit-lakehouse-analytics/
├── config/
│   └── settings.py                   # Paths, subreddit list, model config
├── notebooks/
│   ├── 01_setup_environment.py       # Create directories, verify Spark + Delta
│   ├── 02_bronze_ingestion.py        # Reddit JSON API → Bronze Delta table
│   ├── 03_silver_transformations.py  # Cleaning, dedup, normalization, enrichment
│   ├── 04_gold_aggregations.py       # 5 business-level aggregate tables
│   ├── 05_embedding_generation.py    # Sentence-Transformers → ChromaDB
│   └── 06_run_full_pipeline.py       # Runs all steps end-to-end
├── src/
│   ├── ingestion/
│   │   └── reddit_client.py          # Reddit public JSON fetcher (no API key)
│   ├── transformations/
│   │   ├── bronze_to_silver.py       # PySpark cleaning & enrichment functions
│   │   └── silver_to_gold.py         # Aggregation logic (5 gold tables)
│   └── embeddings/
│       └── generator.py              # Embedding generation + ChromaDB vector store
├── streamlit_app/
│   ├── app.py                        # Streamlit search interface
│   └── search_client.py              # ChromaDB query client
├── data/                             # Generated at runtime (gitignored)
│   ├── bronze/                       # Raw Delta tables
│   ├── silver/                       # Cleaned Delta tables
│   ├── gold/                         # Aggregated Delta tables
│   └── chroma_db/                    # ChromaDB vector store
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies

| Layer | Technology |
|-------|-----------|
| **Data Ingestion** | Reddit public JSON API, `requests` |
| **Processing** | PySpark, Delta Lake (open source) |
| **Storage** | Delta Lake (local, ACID-compliant) |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Vector Search** | ChromaDB (local, persistent) |
| **Frontend** | Streamlit |

## Data Flow

1. **Bronze**: Raw posts fetched from Reddit's public JSON endpoints (no API key needed). Paginated ingestion with rate limiting across 16 subreddits.
2. **Silver**: PySpark transformations — deduplication via windowed row numbering, control character removal, null normalization, boolean casting, and derived feature engineering (content concatenation, temporal features).
3. **Gold**: Five aggregated tables for analytics — subreddit summaries, daily trends, hourly activity patterns, top posts ranking, and author engagement metrics.
4. **Embeddings**: Top 100 posts per subreddit encoded with `all-MiniLM-L6-v2` (384-dim) and stored in ChromaDB with cosine similarity indexing.
5. **Search**: Streamlit app queries ChromaDB for semantic nearest neighbors with optional subreddit filtering.
