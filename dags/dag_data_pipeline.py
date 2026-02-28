"""
DAG 1 — Data Pipeline (hourly)
================================
Schedule : every hour during market hours (Mon–Fri, 14:00–22:00 UTC ~ NYSE hours)
Tasks    :
  1. fetch_sp500_data   — download OHLCV from yfinance
  2. run_feature_eng    — compute RSI, MACD, Bollinger Bands, etc.
  3. save_to_parquet    — persist result to artifacts/data/market/

XCom contract (task → task via return value):
  fetch_sp500_data  → run_feature_eng   : serialized JSON path (str)
  run_feature_eng   → save_to_parquet   : serialized JSON path (str)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make project src available when running inside the Airflow container
_project_root = os.getenv("PYTHONPATH", "/opt/airflow/project")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default task arguments — applied to every task in this DAG
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "hull-tactical",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": False,
    "email_on_failure": False,
    "email_on_retry": False,
}

# ---------------------------------------------------------------------------
# Task implementations
# ---------------------------------------------------------------------------

def _fetch_sp500_data(**ctx) -> str:
    """
    Task 1 — Fetch S&P 500 OHLCV data via yfinance.

    Returns:
        Path (str) of the raw parquet file written to disk.
    """
    from dags.utils.data_utils import fetch_sp500_ohlcv, save_parquet

    df = fetch_sp500_ohlcv(ticker="^GSPC", period_days=365)

    filename = f"sp500_raw_{ctx['ds_nodash']}.parquet"
    path = save_parquet(df, filename)

    logger.info("[fetch] Saved %d rows → %s", len(df), path)
    return str(path)


def _run_feature_engineering(**ctx) -> str:
    """
    Task 2 — Compute financial features on the raw OHLCV data.

    Pulls the raw parquet path from XCom (Task 1), engineers features,
    and saves a new 'enriched' parquet file.

    Returns:
        Path (str) of the features parquet file.
    """
    import pandas as pd
    from dags.utils.data_utils import compute_market_features, save_parquet

    # Retrieve upstream file path
    raw_path: str = ctx["ti"].xcom_pull(task_ids="fetch_sp500_data")
    df_raw = pd.read_parquet(raw_path, engine="pyarrow")

    df_features = compute_market_features(df_raw)

    filename = f"sp500_features_{ctx['ds_nodash']}.parquet"
    path = save_parquet(df_features, filename)

    logger.info("[feature_eng] %d rows, %d features → %s", len(df_features), len(df_features.columns), path)
    return str(path)


def _save_to_parquet(**ctx) -> dict:
    """
    Task 3 — Persist the enriched feature file as the 'latest' snapshot
    used by downstream tasks (model training, RAG, Streamlit).

    Also writes a lightweight metadata JSON alongside the parquet.

    Returns:
        dict with 'path' and 'n_rows' for downstream XCom consumers.
    """
    import shutil
    import pandas as pd
    from dags.utils.data_utils import PARQUET_DIR, ensure_dirs

    ensure_dirs()
    features_path: str = ctx["ti"].xcom_pull(task_ids="run_feature_engineering")
    df = pd.read_parquet(features_path, engine="pyarrow")

    # Write the 'latest' alias used by the training DAG
    latest_path = PARQUET_DIR / "sp500_features_latest.parquet"
    shutil.copy(features_path, latest_path)
    logger.info("[save] Latest snapshot written → %s", latest_path)

    # Metadata sidecar
    meta = {
        "logical_date": ctx["ds"],
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
    }
    meta_path = PARQUET_DIR / "sp500_features_latest_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("[save] Metadata written → %s", meta_path)

    return {"path": str(latest_path), "n_rows": len(df)}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag_data_pipeline",
    description="Hourly: fetch S&P 500 data, engineer features, save to parquet",
    schedule_interval="0 14-22 * * 1-5",   # Every hour Mon–Fri, 14:00–22:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,                      # Prevent overlapping runs
    tags=["data", "market", "yfinance"],
    default_args=DEFAULT_ARGS,
    doc_md=__doc__,
) as dag:

    t1_fetch = PythonOperator(
        task_id="fetch_sp500_data",
        python_callable=_fetch_sp500_data,
        doc_md="Download OHLCV from Yahoo Finance via yfinance.",
    )

    t2_features = PythonOperator(
        task_id="run_feature_engineering",
        python_callable=_run_feature_engineering,
        doc_md="Compute RSI, MACD, Bollinger Bands, rolling stats on OHLCV data.",
    )

    t3_save = PythonOperator(
        task_id="save_to_parquet",
        python_callable=_save_to_parquet,
        doc_md="Persist enriched DataFrame as 'latest' snapshot + metadata JSON.",
    )

    # Pipeline dependency chain
    t1_fetch >> t2_features >> t3_save
