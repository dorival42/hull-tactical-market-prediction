"""
Shared artifact loader for the Hull Tactical Streamlit dashboard.

All pages import from here so there is a single source of truth.
If artifacts/metrics.json or artifacts/feature_importance.json do not exist
(first run before training), sensible fallback defaults are returned.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

KAGGLE_COMPETITION = "hull-tactical-market-prediction"

# ── Root paths ──────────────────────────────────────────────────────────────
# Works from any page depth because we anchor on __file__
_UTILS_DIR = Path(__file__).parent          # app/utils/
_APP_DIR   = _UTILS_DIR.parent              # app/
_ROOT      = _APP_DIR.parent                # project root

ARTIFACTS_DIR = _ROOT / "artifacts"
DATA_PATH     = ARTIFACTS_DIR / "data" / "train.csv"

# ── File paths ───────────────────────────────────────────────────────────────
METRICS_PATH         = ARTIFACTS_DIR / "metrics.json"
FI_PATH              = ARTIFACTS_DIR / "feature_importance.json"
SEL_FEATS_PATH       = ARTIFACTS_DIR / "selected_features.json"
PREPROC_PATH         = ARTIFACTS_DIR / "preprocessing_info.json"
PREPROC_CONFIG_PATH  = ARTIFACTS_DIR / "preprocessing_config.json"

# ── Fallback defaults (used when files not found) ────────────────────────────
_FALLBACK_METRICS: Dict[str, Any] = {
    "LightGBM":     {"rmse": None, "mae": None, "r2": None, "dir_acc": None, "train_time": None, "timestamp": None},
    "XGBoost":      {"rmse": None, "mae": None, "r2": None, "dir_acc": None, "train_time": None, "timestamp": None},
    "CatBoost":     {"rmse": None, "mae": None, "r2": None, "dir_acc": None, "train_time": None, "timestamp": None},
    "RandomForest": {"rmse": None, "mae": None, "r2": None, "dir_acc": None, "train_time": None, "timestamp": None},
    "Ensemble":     {"rmse": None, "mae": None, "r2": None, "dir_acc": None, "train_time": None, "timestamp": None},
}

# ── Loaders (all cached by Streamlit) ────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=60)
def load_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Load per-model metrics from artifacts/metrics.json.

    Returns a dict like::

        {
            "LightGBM": {"rmse": 0.0111, "mae": 0.0079, "r2": 0.004,
                         "dir_acc": 50.45, "train_time": 38.0, "timestamp": "..."},
            ...
        }

    Falls back to None values if the file does not exist (models not trained yet).
    The cache TTL is 60 s so the dashboard refreshes automatically shortly after
    a new training run writes the file.
    """
    if not METRICS_PATH.exists():
        return _FALLBACK_METRICS.copy()
    try:
        with open(METRICS_PATH) as f:
            data = json.load(f)
        # Ensure all expected models are present
        result = _FALLBACK_METRICS.copy()
        result.update(data)
        return result
    except Exception:
        return _FALLBACK_METRICS.copy()


@st.cache_data(show_spinner=False, ttl=60)
def load_feature_importance(model_name: str = "LightGBM") -> pd.DataFrame:
    """
    Load feature importance for *model_name* from artifacts/feature_importance.json.

    Returns a DataFrame with columns [feature, importance, rank] sorted by rank.
    Returns an empty DataFrame if the file does not exist.
    """
    if not FI_PATH.exists():
        return pd.DataFrame(columns=["feature", "importance", "rank"])
    try:
        with open(FI_PATH) as f:
            data = json.load(f)
        if model_name not in data:
            return pd.DataFrame(columns=["feature", "importance", "rank"])
        df = pd.DataFrame(data[model_name])
        return df.sort_values("rank").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance", "rank"])


@st.cache_data(show_spinner=False, ttl=60)
def load_selected_features() -> List[str]:
    """Load selected feature names from artifacts/selected_features.json."""
    if not SEL_FEATS_PATH.exists():
        return []
    try:
        with open(SEL_FEATS_PATH) as f:
            return json.load(f)
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=60)
def load_preprocessing_info() -> Dict[str, Any]:
    """Load preprocessing metadata from artifacts/preprocessing_info.json."""
    if not PREPROC_PATH.exists():
        return {}
    try:
        with open(PREPROC_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _kaggle_download() -> bool:
    """
    Download train.csv from Kaggle using credentials from Streamlit secrets
    or environment variables.  Returns True on success.
    """
    # Resolve credentials: Streamlit secrets > env vars
    try:
        kaggle_username = st.secrets.get("KAGGLE_USERNAME", "") or os.environ.get("KAGGLE_USERNAME", "")
        kaggle_key      = st.secrets.get("KAGGLE_KEY",      "") or os.environ.get("KAGGLE_KEY",      "")
    except Exception:
        kaggle_username = os.environ.get("KAGGLE_USERNAME", "")
        kaggle_key      = os.environ.get("KAGGLE_KEY",      "")

    if not kaggle_username or not kaggle_key:
        return False

    # Set env vars so the kaggle package picks them up
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"]      = kaggle_key

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

        api = KaggleApi()
        api.authenticate()

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        api.competition_download_files(
            competition=KAGGLE_COMPETITION,
            path=str(DATA_PATH.parent),
            quiet=True,
        )

        # Extract zip if present
        zip_path = DATA_PATH.parent / f"{KAGGLE_COMPETITION}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DATA_PATH.parent)
            zip_path.unlink()

        return DATA_PATH.exists()
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_train_data() -> Optional[pd.DataFrame]:
    """
    Load the Kaggle training CSV from artifacts/data/train.csv.

    If the file is absent (e.g. first run on Streamlit Cloud), tries to
    download it automatically using KAGGLE_USERNAME / KAGGLE_KEY credentials
    stored in Streamlit secrets or environment variables.
    """
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    # File not present — attempt download
    with st.spinner("📥 Downloading training data from Kaggle…"):
        success = _kaggle_download()

    if success:
        return pd.read_csv(DATA_PATH)

    return None


# ── Preprocessing config (shared between UI and training pipeline) ────────────

_DEFAULT_PREPROC_CONFIG: Dict[str, Any] = {
    "lower_pct":         0.0,
    "upper_pct":         100.0,
    "nan_threshold_pct": 30,
    "imputation_method": "Median",
    "n_features":        100,
}


def load_preprocessing_config() -> Dict[str, Any]:
    """
    Load user-defined preprocessing config from artifacts/preprocessing_config.json.

    Returns defaults if the file does not exist yet (first run before saving from UI).
    Not cached — always reads fresh so the page picks up changes immediately.
    Falls back to /tmp on Streamlit Cloud where artifacts/ may not be writable.
    """
    _tmp_path = Path("/tmp/preprocessing_config.json")
    for path in [PREPROC_CONFIG_PATH, _tmp_path]:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                result = _DEFAULT_PREPROC_CONFIG.copy()
                result.update(data)
                return result
            except Exception:
                continue
    return _DEFAULT_PREPROC_CONFIG.copy()


def save_preprocessing_config(config: Dict[str, Any]) -> None:
    """
    Persist preprocessing config to artifacts/preprocessing_config.json.

    Called by the preprocessing page when the user clicks "Sauvegarder".
    The trainer (scripts/retrain.py) reads this file on every training run.
    Falls back to /tmp on Streamlit Cloud where /mount/src/ is read-only.
    """
    import datetime as _dt
    payload = dict(config)
    payload["saved_at"] = _dt.datetime.now().isoformat(timespec="seconds")

    # Try artifacts/ first, then /tmp/ as fallback (Streamlit Cloud read-only FS)
    _tmp_path = Path("/tmp/preprocessing_config.json")
    for path in [PREPROC_CONFIG_PATH, _tmp_path]:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            return
        except OSError:
            continue
    raise OSError("Impossible d'écrire la configuration (ni artifacts/ ni /tmp/ accessibles).")


# ── Convenience helpers ──────────────────────────────────────────────────────

def best_model(metrics: Dict[str, Dict[str, Any]], by: str = "rmse") -> str:
    """Return model name with the best (lowest RMSE / highest R²) metric."""
    candidates = {m: v[by] for m, v in metrics.items() if v.get(by) is not None}
    if not candidates:
        return "N/A"
    if by in ("rmse", "mae"):
        return min(candidates, key=lambda k: candidates[k])
    return max(candidates, key=lambda k: candidates[k])


def metrics_as_dataframe(metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Convert the metrics dict to a tidy DataFrame for display."""
    rows = []
    for model, vals in metrics.items():
        rows.append({
            "Model":       model,
            "RMSE":        vals.get("rmse"),
            "MAE":         vals.get("mae"),
            "R²":          vals.get("r2"),
            "Dir. Acc. %": vals.get("dir_acc"),
            "Train Time s":vals.get("train_time"),
            "Timestamp":   vals.get("timestamp"),
        })
    return pd.DataFrame(rows)


def feature_category(feature_name: str) -> str:
    """Classify a feature name into a high-level category."""
    fn = feature_name.lower()
    if "lagged_forward" in fn or "forward_return" in fn:
        return "Forward Returns"
    if fn.startswith("feat_lagged_target") or fn.startswith("lagged_market"):
        return "Lagged Target"
    if fn.startswith("feat_target"):
        return "Target Stats"
    if fn.startswith("feat_v") or (len(fn) <= 4 and fn.startswith("v")):
        return "Volatility"
    if fn.startswith("feat_p") or (len(fn) <= 4 and fn.startswith("p")):
        return "Price"
    if fn.startswith("feat_m") or (len(fn) <= 4 and fn.startswith("m")):
        return "Momentum"
    if fn.startswith("feat_e") or (len(fn) <= 4 and fn.startswith("e")):
        return "Economic"
    if fn.startswith("feat_s") or (len(fn) <= 4 and fn.startswith("s")):
        return "Sentiment"
    if fn.startswith("feat_i") or (len(fn) <= 4 and fn.startswith("i")):
        return "Interest"
    if fn.startswith("feat_"):
        return "Engineered"
    return "Raw"
