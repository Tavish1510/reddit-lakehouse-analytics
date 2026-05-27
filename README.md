# Reddit Lakehouse Analytics & Semantic Search Platform

An end-to-end data pipeline built on **Databricks Lakehouse** that ingests Reddit posts from 16 subreddits, transforms them through a Bronze/Silver/Gold medallion architecture, and powers a **RAG-based Q&A system** using OpenAI embeddings and Databricks Vector Search.

## Architecture

```
┌──────────────┐     ┌───────────────────────────────────────────────────┐     ┌──────────────┐
│  Reddit API  │────▶│              Databricks Lakehouse                 │────▶│  Streamlit   │
│   (PRAW)     │     │                                                   │     │   Q&A App    │
└──────────────┘     │  ┌─────────┐   ┌──────────┐   ┌──────────────┐  │     └──────┬───────┘
                     │  │ BRONZE  │──▶│  SILVER  │──▶│    GOLD      │  │            │
                     │  │ Raw     │   │ Cleaned  │   │ Aggregated   │  │            │
                     │  │ Posts   │   │ Deduped  │   │ + Embeddings │  │            │
                     │  └─────────┘   └──────────┘   └──────┬───────┘  │            │
                     │                                       │          │            │
                     │                              ┌────────▼───────┐  │            │
                     │                              │ Vector Search  │◀─┼────────────┘
                     │                              │    Index       │  │
                     │                              └────────────────┘  │
                     │                                                   │
                     │         Unity Catalog Governance                  │
                     └───────────────────────────────────────────────────┘
```

## Key Features

- **16 Subreddits** ingested daily: technology, programming, MachineLearning, datascience, wallstreetbets, personalfinance, investing, science, space, worldnews, news, AskReddit, explainlikeimfive, fitness, cooking, travel
- **Medallion Architecture**: Bronze (raw) → Silver (cleaned, deduplicated, normalized) → Gold (aggregated analytics + embeddings)
- **PySpark ETL**: Scalable transformations with deduplication, text cleaning, type normalization, and derived feature engineering
- **Delta Lake Optimization**: Partitioning by subreddit, Z-ORDER by date and score, OPTIMIZE commands
- **Unity Catalog**: Full governance with catalog/schema/table hierarchy
- **Semantic Search**: OpenAI `text-embedding-3-small` embeddings stored in Databricks Vector Search
- **RAG Q&A**: Streamlit app using retrieval-augmented generation to answer questions about Reddit content

## Project Structure

```
reddit-lakehouse-analytics/
├── config/
│   └── settings.py              # Configuration constants and env vars
├── notebooks/
│   ├── 01_setup_catalog.py      # Unity Catalog setup (catalog, schemas, tables)
│   ├── 02_bronze_ingestion.py   # Reddit API → Bronze Delta table
│   ├── 03_silver_transformations.py  # Cleaning, dedup, normalization
│   ├── 04_gold_aggregations.py  # Business-level aggregate tables
│   ├── 05_embedding_generation.py    # OpenAI embeddings for top posts
│   └── 06_vector_search_index.py     # Databricks Vector Search setup
├── src/
│   ├── ingestion/
│   │   └── reddit_client.py     # PRAW-based Reddit data fetcher
│   ├── transformations/
│   │   ├── bronze_to_silver.py  # PySpark cleaning & enrichment
│   │   └── silver_to_gold.py    # Aggregation logic (5 gold tables)
│   └── embeddings/
│       └── generator.py         # OpenAI embedding batch generator
├── streamlit_app/
│   ├── app.py                   # Streamlit Q&A interface
│   └── search_client.py         # Databricks Vector Search client
├── .env.example                 # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## Prerequisites

- **Databricks Workspace** (Community Edition works for Bronze/Silver/Gold; Vector Search requires a paid workspace)
- **Reddit API Credentials** — create an app at https://www.reddit.com/prefs/apps
- **OpenAI API Key** — from https://platform.openai.com/api-keys
- Python 3.10+

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Tavish1510/reddit-lakehouse-analytics.git
cd reddit-lakehouse-analytics
```

### 2. Set Up Reddit API Credentials

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app..."
3. Select **script** as the app type
4. Set redirect URI to `http://localhost:8080`
5. Note the `client_id` (under the app name) and `client_secret`

### 3. Configure Databricks Secrets

```bash
# Install the Databricks CLI
pip install databricks-cli

# Configure your workspace
databricks configure --token

# Create a secret scope
databricks secrets create-scope reddit-api

# Store credentials
databricks secrets put-secret reddit-api client-id --string-value "YOUR_REDDIT_CLIENT_ID"
databricks secrets put-secret reddit-api client-secret --string-value "YOUR_REDDIT_CLIENT_SECRET"
databricks secrets put-secret reddit-api openai-key --string-value "YOUR_OPENAI_API_KEY"
```

### 4. Import Notebooks into Databricks

1. In your Databricks workspace, go to **Workspace** → **Users** → your username
2. Right-click → **Import**
3. Upload all `.py` files from the `notebooks/` directory
4. Databricks will recognize them as notebooks automatically

### 5. Run the Pipeline

Execute notebooks in order:
1. `01_setup_catalog` — creates the Unity Catalog structure
2. `02_bronze_ingestion` — fetches Reddit data into Bronze
3. `03_silver_transformations` — cleans and enriches to Silver
4. `04_gold_aggregations` — builds analytics tables in Gold
5. `05_embedding_generation` — generates OpenAI embeddings
6. `06_vector_search_index` — creates the Vector Search index

### 6. Run the Streamlit App

```bash
# Create a .env file from the template
cp .env.example .env
# Edit .env with your credentials

pip install -r requirements.txt
cd streamlit_app
streamlit run app.py
```

## Gold Layer Tables

| Table | Description |
|-------|-------------|
| `subreddit_daily_stats` | Daily post count, avg score, avg comments per subreddit |
| `subreddit_summary` | Overall stats: unique authors, score distribution, NSFW count |
| `hourly_activity` | Post activity by hour and day of week (heatmap data) |
| `top_posts` | Top 50 posts per subreddit by score |
| `author_activity` | Per-author stats by subreddit |
| `post_embeddings` | OpenAI embeddings for semantic search |

## Technologies

- **Databricks Lakehouse** — Delta Lake, Unity Catalog, Vector Search, Workflows
- **PySpark** — distributed data transformations
- **PRAW** — Reddit API wrapper
- **OpenAI API** — text-embedding-3-small for semantic embeddings
- **Streamlit** — interactive Q&A web application
- **Python** — Pandas, NumPy for local data work
