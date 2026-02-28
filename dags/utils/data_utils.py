"""
Shared data helpers used by multiple DAGs.
All functions are pure Python (no Airflow imports) so they can be
unit-tested independently.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.getenv("PYTHONPATH", "/opt/airflow/project"))
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
PARQUET_DIR = DATA_DIR / "market"


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for d in (DATA_DIR, PARQUET_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------
def fetch_sp500_ohlcv(
    ticker: str = "^GSPC",
    period_days: int = 365,
) -> pd.DataFrame:
    """
    Download OHLCV data for the S&P 500 index via yfinance.

    Args:
        ticker: Yahoo Finance ticker symbol (default '^GSPC').
        period_days: Number of calendar days to look back.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume, Adj Close].

    Raises:
        ValueError: If the download returns an empty DataFrame.
    """
    import yfinance as yf

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=period_days)

    logger.info("Fetching %s from %s to %s", ticker, start_date.date(), end_date.date())
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(f"yfinance returned empty DataFrame for ticker '{ticker}'")

    df.index.name = "date"
    df = df.reset_index()
    logger.info("Downloaded %d rows for %s", len(df), ticker)
    return df


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------
def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute financial features on a raw OHLCV DataFrame.

    Features computed (all lag-safe — only use past data):
      - log_return       : daily log return on Adj Close
      - return_5d        : 5-day cumulative return
      - return_20d       : 20-day cumulative return
      - vol_5d           : 5-day rolling std of log_return (annualised)
      - vol_20d          : 20-day rolling std of log_return (annualised)
      - rsi_14           : 14-period RSI
      - macd             : MACD line (12/26 EMA diff)
      - macd_signal      : 9-period EMA of MACD
      - bb_upper / bb_lower : Bollinger Bands (20d, ±2σ)
      - volume_zscore    : z-score of volume over 20 days

    Args:
        df: Raw OHLCV DataFrame from fetch_sp500_ohlcv().

    Returns:
        DataFrame with original columns + new feature columns.
        Rows with NaN (warm-up period) are dropped.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["Close"]

    # --- Returns ---
    df["log_return"] = close.apply(lambda x: x).pct_change().apply(
        lambda r: r  # kept as simple return for compatibility
    )
    # Actually compute as log return
    import numpy as np
    df["log_return"] = np.log(close / close.shift(1))
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)

    # --- Volatility (annualised) ---
    df["vol_5d"] = df["log_return"].rolling(5).std() * np.sqrt(252)
    df["vol_20d"] = df["log_return"].rolling(20).std() * np.sqrt(252)

    # --- RSI (14) ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    # --- Volume z-score ---
    df["volume_zscore"] = (
        (df["Volume"] - df["Volume"].rolling(20).mean())
        / df["Volume"].rolling(20).std()
    )

    df = df.dropna().reset_index(drop=True)
    logger.info("Feature engineering produced %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------
def save_parquet(df: pd.DataFrame, filename: str) -> Path:
    """
    Save a DataFrame to the market parquet directory.

    Args:
        df: DataFrame to persist.
        filename: File name (without directory), e.g. 'sp500_features.parquet'.

    Returns:
        Absolute Path of the written file.
    """
    ensure_dirs()
    path = PARQUET_DIR / filename
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def load_parquet(filename: str) -> Optional[pd.DataFrame]:
    """
    Load a parquet file from the market directory.
    Returns None if the file does not exist.
    """
    path = PARQUET_DIR / filename
    if not path.exists():
        logger.warning("Parquet file not found: %s", path)
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df
