"""
DAG 1 — Data Pipeline (hourly)
================================
Schedule : every hour during market hours (Mon–Fri, 14:00–22:00 UTC ~ NYSE hours)
Tasks    :
  1. fetch_sp500_data       — download OHLCV from yfinance (incremental — HIGH-19)
  2. run_feature_engineering — compute RSI, MACD, Bollinger Bands, rolling stats
  2b. validate_features      — data-quality gate (row count, NaN, required features — MED-17)
  3. save_to_parquet         — persist result to artifacts/data/market/

Fixes applied:
  HIGH-3:  schedule_interval → schedule (Airflow 2.4+ deprecation)
  HIGH-14: on_failure_callback for Slack alerting
  HIGH-15: retry_exponential_backoff=True (1m → 2m → 4m)
  HIGH-19: Incremental yfinance fetch — caches raw OHLCV, fetches only new days
  MED-8:   df["date"] KeyError fixed — handles both DatetimeIndex and column
  MED-17:  Data validation task between feature engineering and save

XCom contract:
  fetch_sp500_data      → run_feature_engineering  : raw parquet path (str)
  run_feature_engineering → validate_features       : features parquet path (str)
  validate_features     → save_to_parquet           : validated features parquet path (str)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

_project_root = os.getenv("PYTHONPATH", "/opt/airflow/project")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from airflow import DAG
from airflow.operators.python import PythonOperator
from utils.alert_utils import on_failure_callback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default task arguments
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "hull-tactical",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,          # HIGH-15: 1m → 2m → 4m backoff
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": False,
    "email_on_retry": False,
    "on_failure_callback": on_failure_callback,  # HIGH-14: Slack + log alerting
}

# ---------------------------------------------------------------------------
# Task implementations
# ---------------------------------------------------------------------------

def _fetch_sp500_data(**ctx) -> str:
    """
    Task 1 — Incremental S&P 500 OHLCV fetch via yfinance (HIGH-19).

    Strategy:
      - Maintains a rolling 365-day raw OHLCV cache (sp500_raw_full.parquet).
      - On each run, fetches only the days since the last cached date (+3 buffer).
      - First run (no cache): downloads full 365-day history.
      - Deduplicates and merges with cache on subsequent runs.

    Returns:
        Path (str) of the raw parquet file written to disk.
    """
    import pandas as pd
    from dags.utils.data_utils import fetch_sp500_ohlcv, save_parquet, PARQUET_DIR

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE = PARQUET_DIR / "sp500_raw_full.parquet"

    # --- Load existing cache ---
    existing_df = None
    if RAW_CACHE.exists():
        try:
            existing_df = pd.read_parquet(RAW_CACHE, engine="pyarrow")
        except Exception as exc:
            logger.warning("[fetch] Cache unreadable (%s) — falling back to full download", exc)

    # --- Determine fetch period ---
    if existing_df is not None and len(existing_df) > 0:
        last_date = pd.to_datetime(existing_df.index.max()).normalize()
        now = pd.Timestamp.today().normalize()
        days_since = max((now - last_date).days, 0)

        if days_since == 0:
            logger.info("[fetch] Data already current (last: %s) — using cache", last_date.date())
            df_combined = existing_df
        else:
            # Fetch only the missing trading days + 3-day buffer for weekends/holidays
            fetch_days = max(days_since + 3, 5)
            logger.info("[fetch] Incremental: %d days since %s", fetch_days, last_date.date())
            df_new = fetch_sp500_ohlcv(ticker="^GSPC", period_days=fetch_days)

            # Merge and deduplicate on DatetimeIndex, keep most recent value
            df_combined = pd.concat([existing_df, df_new])
            df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
            df_combined = df_combined.sort_index()

            # Keep rolling 365-day window
            cutoff = df_combined.index.max() - pd.Timedelta(days=365)
            df_combined = df_combined[df_combined.index >= cutoff]
    else:
        logger.info("[fetch] No cache found — full 365-day download")
        df_combined = fetch_sp500_ohlcv(ticker="^GSPC", period_days=365)

    # --- Persist updated cache ---
    df_combined.to_parquet(RAW_CACHE, engine="pyarrow")

    filename = f"sp500_raw_{ctx['ds_nodash']}.parquet"
    path = save_parquet(df_combined, filename)
    logger.info(
        "[fetch] %d rows (last: %s) → %s", len(df_combined), df_combined.index.max().date(), path
    )
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

    raw_path: str = ctx["ti"].xcom_pull(task_ids="fetch_sp500_data")
    df_raw = pd.read_parquet(raw_path, engine="pyarrow")

    df_features = compute_market_features(df_raw)

    filename = f"sp500_features_{ctx['ds_nodash']}.parquet"
    path = save_parquet(df_features, filename)

    logger.info(
        "[feature_eng] %d rows, %d features → %s", len(df_features), len(df_features.columns), path
    )
    return str(path)


def _validate_features(**ctx) -> str:
    """
    Task 2b — Data-quality gate after feature engineering (MED-17).

    Asserts:
      - Minimum row count (≥ 50) to ensure rolling windows are valid.
      - Maximum NaN ratio (< 80%) to catch partial download failures.
      - Presence of key derived features that indicate compute_market_features ran fully.

    Returns:
        The same features parquet path, passed through to save_to_parquet.
    """
    import pandas as pd

    feat_path: str = ctx["ti"].xcom_pull(task_ids="run_feature_engineering")
    df = pd.read_parquet(feat_path, engine="pyarrow")

    # Row count guard
    assert len(df) >= 50, (
        f"Too few rows after feature engineering: {len(df)} (expected ≥ 50). "
        "yfinance may have returned incomplete data."
    )

    # NaN guard
    max_nan = df.isnull().mean().max()
    assert max_nan < 0.80, (
        f"Excessive NaN in engineered features: {max_nan:.1%} "
        "(threshold: 80%). Check compute_market_features() output."
    )

    # Required feature presence
    required = ["log_return", "rsi_14", "macd", "vol_20d"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, (
        f"Missing required features after engineering: {missing}. "
        "Verify compute_market_features() in data_utils.py."
    )

    logger.info(
        "[validate] Features OK: %d rows, %d cols, max_nan=%.1f%%",
        len(df), len(df.columns), max_nan * 100,
    )
    return feat_path


def _save_to_parquet(**ctx) -> dict:
    """
    Task 3 — Persist the validated feature file as the 'latest' snapshot
    used by downstream tasks (model training, Streamlit).

    Also writes a lightweight metadata JSON alongside the parquet.

    Fixes:
      MED-8: Handles both DatetimeIndex and a 'date' column without KeyError.

    Returns:
        dict with 'path' and 'n_rows' for downstream XCom consumers.
    """
    import shutil
    import pandas as pd
    from dags.utils.data_utils import PARQUET_DIR, ensure_dirs

    ensure_dirs()
    features_path: str = ctx["ti"].xcom_pull(task_ids="validate_features")
    df = pd.read_parquet(features_path, engine="pyarrow")

    # Write the 'latest' alias used by the training DAG
    latest_path = PARQUET_DIR / "sp500_features_latest.parquet"
    shutil.copy(features_path, latest_path)
    logger.info("[save] Latest snapshot written → %s", latest_path)

    # MED-8: Support both DatetimeIndex (yfinance default) and explicit 'date' column
    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"])
    else:
        date_series = pd.to_datetime(df.index.to_series())

    meta = {
        "logical_date": ctx["ds"],
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "date_min": str(date_series.min().date()),
        "date_max": str(date_series.max().date()),
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
    description="Hourly: fetch S&P 500 (incremental), engineer features, validate, save parquet",
    schedule="0 14-22 * * 1-5",            # HIGH-3: schedule_interval deprecated in 2.4+
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
        doc_md="Download OHLCV from Yahoo Finance via yfinance (incremental cache).",
    )

    t2_features = PythonOperator(
        task_id="run_feature_engineering",
        python_callable=_run_feature_engineering,
        doc_md="Compute RSI, MACD, Bollinger Bands, rolling stats on OHLCV data.",
    )

    t2b_validate = PythonOperator(
        task_id="validate_features",
        python_callable=_validate_features,
        doc_md="Data-quality gate: assert row count, NaN ratio, required feature presence.",
    )

    t3_save = PythonOperator(
        task_id="save_to_parquet",
        python_callable=_save_to_parquet,
        doc_md="Persist validated features as 'latest' snapshot + metadata JSON.",
    )

    # Pipeline dependency chain: fetch → engineer → validate → save
    t1_fetch >> t2_features >> t2b_validate >> t3_save
