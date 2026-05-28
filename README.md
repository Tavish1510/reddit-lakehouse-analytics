# Reddit Lakehouse Analytics

End-to-end data pipeline on **PySpark + Delta Lake** that ingests Reddit posts from 16 subreddits through a Bronze / Silver / Gold medallion architecture, with a **semantic search** layer built on Sentence-Transformers and ChromaDB.

**Live demo:** https://reddit-lakehouse-analytics-kjcdrfnrdfvxkwe3tsqvu5.streamlit.app/

## Architecture

```
┌──────────────────┐    ┌──────────────────────────────────────────┐    ┌──────────────┐
│ Reddit JSON API  │───▶│         Local Delta Lakehouse            │───▶│  Streamlit   │
│ (no auth)        │    │  Bronze ──▶ Silver ──▶ Gold (5 tables)   │    │ Search App   │
└──────────────────┘    └──────────────────────────────────────────┘    └──────┬───────┘
                                                                                │
                        ┌──────────────────────────────────────────┐            │
                        │ Sentence-Transformers (all-MiniLM-L6-v2) │◀───────────┘
                        │             →  ChromaDB                  │
                        └──────────────────────────────────────────┘
```

## Highlights

- **Medallion architecture** with PySpark transformations: deduplication via windowed `row_number`, control-char cleanup, null/boolean normalization, derived temporal features.
- **Delta Lake** for ACID guarantees, schema enforcement, partitioning by `subreddit`.
- **Five Gold tables**: `subreddit_summary`, `subreddit_daily_stats`, `hourly_activity`, `top_posts`, `author_activity`.
- **Semantic search** over the top 100 posts/subreddit (1,600 vectors, 384-dim) with cosine similarity.
- **No API keys, no cloud accounts** — Reddit's public JSON endpoints + local Spark + local ChromaDB.

## Stack

PySpark · Delta Lake · ChromaDB · Sentence-Transformers · Streamlit · Python

## Project layout

```
config/settings.py
src/
  ingestion/reddit_client.py       # paginated JSON fetcher
  transformations/                 # bronze→silver, silver→gold (pure PySpark fns)
  embeddings/generator.py
notebooks/                         # 01–06, runnable end-to-end
streamlit_app/
  app.py
  search_client.py
data/                              # generated at runtime (Delta + ChromaDB)
```

## Quick start

```bash
git clone https://github.com/Tavish1510/reddit-lakehouse-analytics.git
cd reddit-lakehouse-analytics
python -m venv .venv && source .venv/bin/activate    # or .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python notebooks/06_run_full_pipeline.py             # bronze → silver → gold → embeddings
streamlit run streamlit_app/app.py
```

Requires **Java 11 or 17** for PySpark. On Windows, `winutils.exe` / `hadoop.dll` need to be present and `HADOOP_HOME` set — see [cdarlint/winutils](https://github.com/cdarlint/winutils) for the standard Windows fix.

## Gold tables

| Table | Description |
|---|---|
| `subreddit_summary` | Overall stats per subreddit (post count, score percentiles, unique authors) |
| `subreddit_daily_stats` | Daily post volume, avg score, avg comments |
| `hourly_activity` | Posts × hour × day-of-week (heatmap-ready) |
| `top_posts` | Top 50 posts per subreddit by score |
| `author_activity` | Per-author totals and averages |

## Subreddits ingested

technology · programming · MachineLearning · datascience · wallstreetbets · personalfinance · investing · science · space · worldnews · news · AskReddit · explainlikeimfive · fitness · cooking · travel
