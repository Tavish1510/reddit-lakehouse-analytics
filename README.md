# Reddit Lakehouse Analytics & Semantic Search Platform

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

## Quick Start

### Prerequisites
- Python 3.10+
- Java 11 or 17 (required by PySpark — [download OpenJDK](https://adoptium.net/))
- **Windows only:** `winutils.exe` + `hadoop.dll` for Hadoop on Windows

### Setup (Windows)

```powershell
# 1. Install Java (OpenJDK 17) and set JAVA_HOME
winget install Microsoft.OpenJDK.17
[Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot", "User")

# 2. Set up winutils (Hadoop on Windows)
New-Item -ItemType Directory -Path "C:\hadoop\bin" -Force
$base = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin"
Invoke-WebRequest "$base/winutils.exe" -OutFile "C:\hadoop\bin\winutils.exe"
Invoke-WebRequest "$base/hadoop.dll" -OutFile "C:\hadoop\bin\hadoop.dll"
[Environment]::SetEnvironmentVariable("HADOOP_HOME", "C:\hadoop", "User")

# 3. Close & reopen PowerShell so the env vars take effect

# 4. Clone and set up the project
git clone https://github.com/Tavish1510/reddit-lakehouse-analytics.git
cd reddit-lakehouse-analytics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Setup (Mac/Linux)

```bash
# Install Java via your package manager (e.g., brew install openjdk@17)

git clone https://github.com/Tavish1510/reddit-lakehouse-analytics.git
cd reddit-lakehouse-analytics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
# Run the full pipeline (ingestion → transformations → aggregations → embeddings)
python notebooks/06_run_full_pipeline.py

# Launch the Streamlit search app
cd streamlit_app
streamlit run app.py
```

Or run steps individually:

```bash
python notebooks/01_setup_environment.py    # ~5 sec
python notebooks/02_bronze_ingestion.py     # ~5 min (fetching from Reddit)
python notebooks/03_silver_transformations.py   # ~30 sec
python notebooks/04_gold_aggregations.py    # ~30 sec
python notebooks/05_embedding_generation.py # ~2 min (first run downloads model)
```

## Gold Layer Tables

| Table | Description |
|-------|-------------|
| `subreddit_daily_stats` | Daily post count, avg score, avg comments per subreddit |
| `subreddit_summary` | Overall stats: unique authors, score distribution, NSFW count |
| `hourly_activity` | Post activity by hour and day of week (heatmap data) |
| `top_posts` | Top 50 posts per subreddit by score |
| `author_activity` | Per-author posting stats by subreddit |

## Deploy to Streamlit Community Cloud

The Streamlit app can be deployed for free at [share.streamlit.io](https://share.streamlit.io):

1. Push this repo to GitHub
2. Sign in to Streamlit Community Cloud with your GitHub account
3. Click **"New app"** → select the `reddit-lakehouse-analytics` repo
4. Set:
   - **Main file path:** `streamlit_app/app.py`
   - **Requirements file:** `streamlit_app/requirements.txt` (lightweight, no PySpark)
5. Click **Deploy**

The `data/chroma_db/` directory is committed to the repo so the deployed app has the pre-built semantic index. To refresh with new posts, rerun the pipeline locally and commit the updated `data/chroma_db/`.

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
