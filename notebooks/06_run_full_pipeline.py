"""
06 — Full Pipeline Runner
Runs the complete pipeline end-to-end in a single script:
  Bronze ingestion -> Silver transformations -> Gold aggregations -> Embeddings

Usage:
    python notebooks/06_run_full_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent
STEPS = [
    ("01 — Setup Environment", "01_setup_environment.py"),
    ("02 — Bronze Ingestion", "02_bronze_ingestion.py"),
    ("03 — Silver Transformations", "03_silver_transformations.py"),
    ("04 — Gold Aggregations", "04_gold_aggregations.py"),
    ("05 — Embedding Generation", "05_embedding_generation.py"),
]


def run_step(name: str, script: str):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")
    start = time.time()

    result = subprocess.run(
        [sys.executable, str(NOTEBOOKS_DIR / script)],
        capture_output=False,
    )

    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit code {result.returncode})")
        sys.exit(1)

    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    total_start = time.time()

    for name, script in STEPS:
        run_step(name, script)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — {total_elapsed:.1f}s total")
    print(f"{'=' * 60}")
    print(f"\nRun the Streamlit app:")
    print(f"  cd streamlit_app && streamlit run app.py")
