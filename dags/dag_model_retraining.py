"""
DAG 3 — Model Retraining (weekly)
=====================================
Schedule : every Monday at 06:00 UTC (after weekend market close)
Tasks    :
  0. validate_mlflow_config  — guard: verify MLFLOW_TRACKING_URI
  1. load_training_data      — Kaggle CSV (preferred) OR yfinance parquet (fallback)
  2. run_preprocessing       — DataPipeline combined method (consistent with manual pipeline)
  3a. train_lightgbm  \
  3b. train_xgboost    | parallel tasks — run simultaneously on LocalExecutor
  3c. train_catboost   |
  3d. train_random_forest /
  4. train_ensemble          — weighted average; saves ensemble_final.pkl
  5. evaluate_models         — RMSE/MAE/R²/DirAcc → metrics.json (read by Streamlit)
  6. promote_if_better       — MLflow Registry → Production if RMSE improves
  7. cleanup_splits          — remove temporary per-run parquet files

Fixes applied vs. original:
  CRIT-2:  ALLOW_LOCAL_MLFLOW respected in validate_mlflow_config
  CRIT-4:  Artifact filenames aligned with Streamlit (*_final.pkl, not *_airflow.pkl)
  CRIT-5:  Ensemble persisted as ensemble_final.pkl in load_model_pkl() format
  HIGH-3:  schedule_interval → schedule
  HIGH-6:  _run_preprocessing uses DataPipeline (combined method) not variance-only
  HIGH-14: on_failure_callback for Slack + log alerting
  HIGH-15: retry_exponential_backoff=True
  HIGH-18: 4 model training tasks run in parallel
  MED-7:   cleanup_splits task removes temporary files after each run
  MED-9:   load_training_data prioritises Kaggle CSV (has proper target variable)

MLflow experiment: 'hull-tactical-experiments' (same as manual pipeline)
Model registry name: 'hull-tactical-best'
Artifact names: *_final.pkl — compatible with app/utils/artifacts.py _MODEL_FILE_MAP
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_project_root = os.getenv("PYTHONPATH", "/opt/airflow/project")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from airflow import DAG
from airflow.operators.python import PythonOperator
from dags.utils.alert_utils import on_failure_callback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "hull-tactical",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,          # HIGH-15: 1m → 2m → 4m
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": False,
    "email_on_retry": False,
    "on_failure_callback": on_failure_callback,  # HIGH-14: Slack + log alerting
}

ARTIFACTS_DIR = Path(_project_root) / "artifacts"
DATA_DIR      = ARTIFACTS_DIR / "data"
PARQUET_DIR   = DATA_DIR / "market"
REGISTERED_MODEL_NAME = "hull-tactical-best"

# CRIT-4: Canonical artifact filenames — must match app/utils/artifacts.py _MODEL_FILE_MAP
_ARTIFACT_FILENAMES = {
    "LightGBM":     "lightgbm_final.pkl",
    "XGBoost":      "xgboost_final.pkl",
    "CatBoost":     "catboost_final.pkl",
    "RandomForest": "random_forest_final.pkl",
    "Ensemble":     "ensemble_final.pkl",     # CRIT-5: now actually written
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save_model_pkl(model_obj: Any, name: str, feature_names: list, path: str) -> None:
    """Persist model in BaseModel-compatible dict format for load_model_pkl()."""
    with open(path, "wb") as f:
        pickle.dump({
            "name": name,
            "params": {},
            "model": model_obj,
            "feature_names": feature_names,
            "training_metrics": {},
        }, f)


# ---------------------------------------------------------------------------
# Task 0 — Validate MLflow config
# ---------------------------------------------------------------------------

def _validate_mlflow_config(**ctx) -> dict:
    """Guard: MLFLOW_TRACKING_URI must be set; DagsHub required unless ALLOW_LOCAL_MLFLOW=true."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI is not set. "
            "Set it in docker-compose.airflow.yml or Airflow Variables."
        )
    if "dagshub" not in uri and os.getenv("ALLOW_LOCAL_MLFLOW", "false").lower() != "true":
        raise ValueError(
            f"MLFLOW_TRACKING_URI='{uri}' does not point to DagsHub. "
            "Set ALLOW_LOCAL_MLFLOW=true to allow local tracking (already set in docker-compose)."
        )
    logger.info("[validate] MLflow URI OK: %s", uri)
    return {"mlflow_uri": uri}


# ---------------------------------------------------------------------------
# Task 1 — Load training data
# ---------------------------------------------------------------------------

def _load_training_data(**ctx) -> dict:
    """
    Assemble training dataset with explicit data-source priority (MED-9).

    Priority:
      1. Kaggle CSV (has proper target: market_forward_excess_returns)
      2. yfinance parquet (fallback: uses log_return as target)

    The two sources are NOT concatenated because they have incompatible schemas.
    """
    import hashlib
    import pandas as pd

    TARGET_COL = "market_forward_excess_returns"
    frames_with_target: list[pd.DataFrame] = []
    frames_without_target: list[pd.DataFrame] = []

    # Source 1 — Kaggle competition CSV (preferred: proper target variable)
    kaggle_csv = DATA_DIR / "train.csv"
    if kaggle_csv.exists():
        df_k = pd.read_csv(kaggle_csv)
        logger.info("[load] Kaggle CSV: %d rows, %d cols", len(df_k), len(df_k.columns))
        (frames_with_target if TARGET_COL in df_k.columns else frames_without_target).append(df_k)
    else:
        logger.warning("[load] Kaggle CSV not found — %s", kaggle_csv)

    # Source 2 — yfinance enriched parquet (fallback: log_return as target)
    latest_parquet = PARQUET_DIR / "sp500_features_latest.parquet"
    if latest_parquet.exists():
        df_m = pd.read_parquet(latest_parquet, engine="pyarrow")
        logger.info("[load] Market parquet: %d rows, %d cols", len(df_m), len(df_m.columns))
        (frames_with_target if TARGET_COL in df_m.columns else frames_without_target).append(df_m)
    else:
        logger.warning("[load] Market parquet not found — %s", latest_parquet)

    if not frames_with_target and not frames_without_target:
        raise FileNotFoundError(
            "No training data found. Run dag_data_pipeline first, "
            "or place train.csv in artifacts/data/."
        )

    # Select: prefer source with proper target variable
    if frames_with_target:
        df = frames_with_target[0]
        logger.info("[load] Using source WITH target column (%d rows)", len(df))
    else:
        df = frames_without_target[0]
        logger.info("[load] Fallback to source WITHOUT target (%d rows) — will use log_return", len(df))

    # Data hash for MLflow reproducibility tracking
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:12]

    out_path = DATA_DIR / f"training_snapshot_{ctx['ds_nodash']}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    return {
        "path": str(out_path),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "data_hash": data_hash,
    }


# ---------------------------------------------------------------------------
# Task 2 — Preprocessing (HIGH-6: canonical DataPipeline combined method)
# ---------------------------------------------------------------------------

def _run_preprocessing(**ctx) -> dict:
    """
    Preprocess using the canonical DataPipeline (combined method: HIGH-6).

    Uses the same DataPipeline(feature_selection_method='combined') as the manual
    training pipeline, producing consistent feature sets across both code paths.
    Combined scoring = 50% importance + 25% correlation + 25% mutual_info.
    """
    import pandas as pd
    from src.data.preprocessor import DataPipeline

    task_data: dict = ctx["ti"].xcom_pull(task_ids="load_training_data")
    df = pd.read_parquet(task_data["path"], engine="pyarrow")

    target_col = "market_forward_excess_returns"
    if target_col not in df.columns:
        if "log_return" in df.columns:
            target_col = "log_return"
            logger.info("[preprocess] Using 'log_return' as target (yfinance fallback).")
        else:
            raise ValueError("No usable target column found in training data.")

    # Canonical DataPipeline — same as MLflowTrainer.load_and_prepare_data()
    pipeline = DataPipeline(
        nan_threshold=0.30,
        use_catboost_imputation=False,      # Median imputation (matches artifacts)
        n_features=100,
        feature_selection_method="combined",
        verbose=False,
    )
    pipeline.fit(df, target_col=target_col)
    selected = pipeline.get_selected_features()
    df_processed = pipeline.preprocessor.transform(df)

    y = df_processed[target_col].dropna()
    X = df_processed.loc[y.index, selected]

    # Time-ordered 80/20 split (no random shuffle — time series)
    split_idx = int(len(X) * 0.80)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    out_dir = DATA_DIR / f"splits_{ctx['ds_nodash']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(out_dir / "X_train.parquet", index=False)
    X_val.to_parquet(out_dir / "X_val.parquet",   index=False)
    y_train.to_frame().to_parquet(out_dir / "y_train.parquet", index=False)
    y_val.to_frame().to_parquet(out_dir / "y_val.parquet",   index=False)

    # Airflow-specific feature list (does not overwrite manual pipeline output)
    (ARTIFACTS_DIR / "selected_features_airflow.json").write_text(json.dumps(selected))

    logger.info(
        "[preprocess] Train: %d | Val: %d | Features: %d (combined method)",
        len(X_train), len(X_val), len(selected),
    )
    return {
        "splits_dir":    str(out_dir),
        "n_train":       len(X_train),
        "n_val":         len(X_val),
        "n_features":    len(selected),
        "target_col":    target_col,
        "nan_threshold": 0.30,
        "feature_cols":  selected,
    }


# ---------------------------------------------------------------------------
# Tasks 3a-3d — Individual model training (HIGH-18: parallel PythonOperators)
# ---------------------------------------------------------------------------

def _make_train_fn(model_key: str, display_name: str):
    """
    Factory that returns a self-contained training callable for one model type.

    Each returned function is a standalone PythonOperator callable that:
      - Reads splits from run_preprocessing XCom
      - Loads hyperparameters from Config (config.yaml + model_params.yaml)
      - Trains the model with early stopping
      - Saves to *_final.pkl (CRIT-4: Streamlit-compatible filename)
      - Returns result dict or None on failure (task does not fail the DAG)
    """
    def _train(**ctx) -> dict | None:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import mean_squared_error
        from src.utils.config import Config

        prep_data: dict = ctx["ti"].xcom_pull(task_ids="run_preprocessing")
        splits_dir = Path(prep_data["splits_dir"])
        X_train = pd.read_parquet(splits_dir / "X_train.parquet")
        X_val   = pd.read_parquet(splits_dir / "X_val.parquet")
        y_train = pd.read_parquet(splits_dir / "y_train.parquet").squeeze()
        y_val   = pd.read_parquet(splits_dir / "y_val.parquet").squeeze()

        config = Config()
        params = config.get_model_params(model_key)
        artifact_path = str(ARTIFACTS_DIR / _ARTIFACT_FILENAMES[display_name])

        try:
            if model_key == "lightgbm":
                import lightgbm as lgb
                m = lgb.LGBMRegressor(**params)
                m.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
                )

            elif model_key == "xgboost":
                import xgboost as xgb
                m = xgb.XGBRegressor(**params)
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            elif model_key == "catboost":
                from catboost import CatBoostRegressor
                early_stopping = params.pop("early_stopping_rounds", 50)
                m = CatBoostRegressor(**params)
                m.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=early_stopping,
                )

            elif model_key == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                m = RandomForestRegressor(**params)
                m.fit(X_train, y_train)

            else:
                raise ValueError(f"Unknown model_key: {model_key}")

            preds = m.predict(X_val)
            rmse  = float(np.sqrt(mean_squared_error(y_val, preds)))
            _save_model_pkl(m, display_name, X_train.columns.tolist(), artifact_path)
            logger.info("[train] %s RMSE=%.6f → %s", display_name, rmse, artifact_path)
            return {
                "model_name":    display_name,
                "rmse":          rmse,
                "artifact_path": artifact_path,
                "params":        params,
            }

        except Exception as exc:
            logger.error("[train] %s FAILED: %s", display_name, exc, exc_info=True)
            return None   # Allow other parallel tasks to continue

    _train.__name__ = f"_train_{model_key}"
    return _train


_train_lightgbm     = _make_train_fn("lightgbm",     "LightGBM")
_train_xgboost      = _make_train_fn("xgboost",      "XGBoost")
_train_catboost     = _make_train_fn("catboost",      "CatBoost")
_train_random_forest = _make_train_fn("random_forest", "RandomForest")


# ---------------------------------------------------------------------------
# Task 4 — Ensemble (CRIT-5: now saves ensemble_final.pkl)
# ---------------------------------------------------------------------------

def _train_ensemble(**ctx) -> dict | None:
    """
    Build a weighted ensemble from all successfully trained sub-models (CRIT-5).

    Saves ensemble_final.pkl in the nested format expected by load_model_pkl():
      {"models": [{"name": ..., "model": ..., "feature_names": ...}],
       "weights": {"LightGBM": 0.35, ...},
       "feature_names": [...]}

    Runs AFTER all 4 parallel training tasks (triggered by Airflow dependency graph).
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from src.utils.config import Config

    prep_data: dict = ctx["ti"].xcom_pull(task_ids="run_preprocessing")
    splits_dir = Path(prep_data["splits_dir"])
    X_val = pd.read_parquet(splits_dir / "X_val.parquet")
    y_val = pd.read_parquet(splits_dir / "y_val.parquet").squeeze()

    config = Config()
    ensemble_cfg = config.model_params.get("ensemble", {})
    weights_cfg: dict = dict(ensemble_cfg.get("weights", {}))

    TASK_MAP = [
        ("train_lightgbm",     "LightGBM"),
        ("train_xgboost",      "XGBoost"),
        ("train_catboost",     "CatBoost"),
        ("train_random_forest", "RandomForest"),
    ]

    sub_models: list[dict] = []
    weighted_sum: np.ndarray | None = None
    total_weight = 0.0

    for task_id, display_name in TASK_MAP:
        result = ctx["ti"].xcom_pull(task_ids=task_id)
        if result is None:
            logger.warning("[ensemble] %s was skipped or failed — excluded", display_name)
            continue

        pkl_path = result["artifact_path"]
        if not Path(pkl_path).exists():
            logger.warning("[ensemble] Artifact missing for %s: %s", display_name, pkl_path)
            continue

        with open(pkl_path, "rb") as f:
            model_data = pickle.load(f)

        raw_model = model_data["model"]
        feat_names = model_data.get("feature_names", X_val.columns.tolist())
        X_input = X_val[feat_names] if feat_names else X_val

        preds = raw_model.predict(X_input)
        w = weights_cfg.get(display_name.lower(), 1.0)

        sub_models.append({"name": display_name, "model": raw_model, "feature_names": feat_names})
        weighted_sum = (weighted_sum + w * preds) if weighted_sum is not None else w * preds
        total_weight += w

    if not sub_models:
        logger.error("[ensemble] No sub-models available — skipping ensemble.")
        return None

    ensemble_preds = weighted_sum / total_weight
    ens_rmse = float(np.sqrt(mean_squared_error(y_val, ensemble_preds)))

    # Normalized weights for the saved pkl
    norm_weights = {
        sub["name"]: weights_cfg.get(sub["name"].lower(), 1.0) / total_weight
        for sub in sub_models
    }

    # CRIT-5: Save in the format Streamlit's _get_predictions() expects for Ensemble
    ens_path = str(ARTIFACTS_DIR / _ARTIFACT_FILENAMES["Ensemble"])
    with open(ens_path, "wb") as f:
        pickle.dump({
            "models":        sub_models,
            "weights":       norm_weights,
            "feature_names": X_val.columns.tolist(),
        }, f)

    logger.info(
        "[ensemble] RMSE=%.6f (%d sub-models, weights=%s) → %s",
        ens_rmse, len(sub_models), norm_weights, ens_path,
    )
    return {
        "model_name":    "Ensemble",
        "rmse":          ens_rmse,
        "artifact_path": ens_path,
        "params":        {},
    }


# ---------------------------------------------------------------------------
# Task 5 — Evaluate all models → metrics.json
# ---------------------------------------------------------------------------

def _evaluate_models(**ctx) -> dict:
    """
    Compute full evaluation metrics for every trained model and persist metrics.json.

    Pulls XCom results from each of the 5 training tasks individually.
    Metrics: RMSE, MAE, R², Directional Accuracy (as %) — consistent with
    ModelMetrics.calculate_all_metrics() in the manual pipeline.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    prep_data: dict = ctx["ti"].xcom_pull(task_ids="run_preprocessing")
    splits_dir = Path(prep_data["splits_dir"])
    X_val = pd.read_parquet(splits_dir / "X_val.parquet")
    y_val = pd.read_parquet(splits_dir / "y_val.parquet").squeeze()

    # Collect results from all parallel training tasks + ensemble
    TASK_NAMES = [
        ("train_lightgbm",     "LightGBM"),
        ("train_xgboost",      "XGBoost"),
        ("train_catboost",     "CatBoost"),
        ("train_random_forest", "RandomForest"),
        ("train_ensemble",     "Ensemble"),
    ]
    all_results: dict[str, dict] = {}
    for task_id, display_name in TASK_NAMES:
        r = ctx["ti"].xcom_pull(task_ids=task_id)
        if r is not None:
            all_results[display_name] = r

    metrics_by_model: dict = {}
    best_single_model: str | None = None
    best_rmse = float("inf")

    for model_name, info in all_results.items():
        artifact_path = info.get("artifact_path", "")

        # --- Ensemble: re-run inference for complete metrics ---
        if model_name == "Ensemble":
            if artifact_path and Path(artifact_path).exists():
                with open(artifact_path, "rb") as f:
                    ens_data = pickle.load(f)
                weights_map = ens_data.get("weights", {})
                sub_preds_list, sub_weights_list = [], []
                for sub in ens_data.get("models", []):
                    fn = sub.get("feature_names", X_val.columns.tolist())
                    Xi = X_val[fn] if fn else X_val
                    sub_preds_list.append(sub["model"].predict(Xi))
                    sub_weights_list.append(weights_map.get(sub["name"], 1.0))
                if sub_preds_list:
                    w_arr = np.array(sub_weights_list)
                    preds = np.average(np.array(sub_preds_list), axis=0, weights=w_arr)
                    metrics_by_model["Ensemble"] = {
                        "rmse":       float(np.sqrt(mean_squared_error(y_val, preds))),
                        "mae":        float(mean_absolute_error(y_val, preds)),
                        "r2":         float(r2_score(y_val, preds)),
                        "dir_acc":    float(np.mean(np.sign(preds) == np.sign(y_val.values)) * 100),
                        "train_time": None,
                        "timestamp":  datetime.utcnow().isoformat(timespec="seconds"),
                    }
            continue

        # --- Individual model ---
        if not artifact_path or not Path(artifact_path).exists():
            logger.warning("[eval] Artifact missing for %s: %s", model_name, artifact_path)
            continue

        with open(artifact_path, "rb") as f:
            model_data = pickle.load(f)

        raw_model = model_data.get("model", model_data) if isinstance(model_data, dict) else model_data
        feat_names = model_data.get("feature_names") if isinstance(model_data, dict) else None
        X_input = X_val[feat_names] if feat_names else X_val

        preds   = raw_model.predict(X_input)
        rmse    = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae     = float(mean_absolute_error(y_val, preds))
        r2      = float(r2_score(y_val, preds))
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y_val.values)) * 100)

        metrics_by_model[model_name] = {
            "rmse":       rmse,
            "mae":        mae,
            "r2":         r2,
            "dir_acc":    dir_acc,
            "train_time": None,
            "timestamp":  datetime.utcnow().isoformat(timespec="seconds"),
        }

        if rmse < best_rmse:
            best_rmse = rmse
            best_single_model = model_name

    # Check if Ensemble is overall winner
    ens_rmse = (metrics_by_model.get("Ensemble") or {}).get("rmse")
    best_overall = best_single_model
    best_overall_rmse = best_rmse
    if ens_rmse is not None and ens_rmse < best_rmse:
        best_overall = "Ensemble"
        best_overall_rmse = ens_rmse

    # Persist metrics.json — read by Streamlit dashboard (60s cache TTL)
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_by_model, indent=2))
    logger.info(
        "[eval] metrics.json written. Best single: %s (%.6f) | Best overall: %s (%.6f)",
        best_single_model, best_rmse, best_overall, best_overall_rmse,
    )

    return {
        "metrics":         metrics_by_model,
        "best_model":      best_single_model,   # single model eligible for registry
        "best_overall":    best_overall,         # may be Ensemble (informational)
        "best_rmse":       best_overall_rmse,
        "all_results":     all_results,
    }


# ---------------------------------------------------------------------------
# Task 6 — MLflow promotion
# ---------------------------------------------------------------------------

def _promote_if_better(**ctx) -> dict:
    """
    Log best single model to MLflow and promote to Production if RMSE improves.

    Only single models (not Ensemble) are registered — Ensemble has no single
    sklearn artifact compatible with mlflow.sklearn.log_model().
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    from dags.utils.mlflow_utils import get_or_create_experiment, register_model_if_better

    eval_data:  dict = ctx["ti"].xcom_pull(task_ids="evaluate_models")
    prep_data:  dict = ctx["ti"].xcom_pull(task_ids="run_preprocessing")
    load_data:  dict = ctx["ti"].xcom_pull(task_ids="load_training_data")

    best_model_name: str | None = eval_data["best_model"]
    metrics: dict               = eval_data["metrics"]
    all_results: dict           = eval_data["all_results"]

    if best_model_name is None:
        logger.warning("[promote] No single model trained successfully — skipping registry.")
        return {"promoted": False, "reason": "no single best model"}

    artifact_path = all_results[best_model_name]["artifact_path"]
    if not Path(artifact_path).exists():
        logger.warning("[promote] Artifact missing: %s", artifact_path)
        return {"promoted": False, "reason": "artifact missing"}

    with open(artifact_path, "rb") as f:
        model_data = pickle.load(f)

    raw_model  = model_data.get("model", model_data) if isinstance(model_data, dict) else model_data
    feat_names = model_data.get("feature_names") if isinstance(model_data, dict) else None

    import pandas as pd
    splits_dir = Path(prep_data["splits_dir"])
    X_val  = pd.read_parquet(splits_dir / "X_val.parquet")
    X_input = X_val[feat_names] if feat_names else X_val

    import numpy as np
    sample_preds = raw_model.predict(X_input.head(5))
    signature    = infer_signature(X_input.head(5), sample_preds)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    exp_id = get_or_create_experiment()

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"airflow-weekly-{ctx['ds']}-{best_model_name}",
        tags={
            "dag":           "dag_model_retraining",
            "triggered_by":  "airflow",
            "best_model":    best_model_name,
        },
    ) as run:
        mlflow.log_params({
            "model_type":               best_model_name,
            "feature_selection_method": "combined",
            "n_train":                  prep_data["n_train"],
            "n_val":                    prep_data["n_val"],
            "n_features":               prep_data["n_features"],
            "nan_threshold":            prep_data["nan_threshold"],
            "target_col":               prep_data["target_col"],
            "logical_date":             ctx["ds"],
            "data_hash":                load_data.get("data_hash", "unknown"),
        })

        model_params = all_results[best_model_name].get("params", {})
        if model_params:
            mlflow.log_params({f"hp_{k}": v for k, v in model_params.items()})

        # Per-model metrics with prefix (UI readability)
        for name, m in metrics.items():
            for k, v in m.items():
                if v is not None and k != "timestamp":
                    mlflow.log_metric(f"{name.lower()}_{k}", float(v))

        # CRITICAL: bare 'rmse' key required by register_model_if_better() comparison
        mlflow.log_metric("rmse",    metrics[best_model_name]["rmse"])
        mlflow.log_metric("dir_acc", metrics[best_model_name].get("dir_acc") or 0.0)

        # Feature list artifact for audit trail
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(feat_names or [], tf)
            mlflow.log_artifact(tf.name, artifact_path="feature_lists")

        mlflow.sklearn.log_model(
            raw_model,
            artifact_path="model",
            signature=signature,
            input_example=X_input.head(5),
            registered_model_name=None,   # register explicitly below for conditional promotion
        )
        run_id = run.info.run_id

    promoted = register_model_if_better(
        run_id=run_id,
        model_artifact_path="model",
        model_name=REGISTERED_MODEL_NAME,
        metric_name="rmse",
        new_metric_value=metrics[best_model_name]["rmse"],
        lower_is_better=True,
    )

    result = {
        "promoted":   promoted,
        "best_model": best_model_name,
        "best_rmse":  metrics[best_model_name]["rmse"],
        "run_id":     run_id,
    }
    logger.info("[promote] Result: %s", result)
    return result


# ---------------------------------------------------------------------------
# Task 7 — Cleanup temporary splits (MED-7)
# ---------------------------------------------------------------------------

def _cleanup_splits(**ctx) -> dict:
    """Remove per-run split parquets and stale training snapshots (MED-7)."""
    prep_data: dict = ctx["ti"].xcom_pull(task_ids="run_preprocessing")
    splits_dir = Path(prep_data["splits_dir"])
    shutil.rmtree(splits_dir, ignore_errors=True)

    # Remove training snapshots older than the current run date
    current_ds = ctx["ds_nodash"]
    removed = 0
    for snapshot in DATA_DIR.glob("training_snapshot_*.parquet"):
        if current_ds not in snapshot.name:
            snapshot.unlink(missing_ok=True)
            removed += 1

    logger.info("[cleanup] Removed splits dir + %d old snapshot(s)", removed)
    return {"cleaned_splits": str(splits_dir), "old_snapshots_removed": removed}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag_model_retraining",
    description="Weekly: retrain 4 models in parallel, ensemble, evaluate, promote to MLflow Production",
    schedule="0 6 * * 1",              # HIGH-3: schedule_interval deprecated in Airflow 2.4+
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["training", "mlflow", "weekly"],
    default_args=DEFAULT_ARGS,
    doc_md=__doc__,
) as dag:

    t0_validate = PythonOperator(
        task_id="validate_mlflow_config",
        python_callable=_validate_mlflow_config,
        doc_md="Guard: verify MLFLOW_TRACKING_URI is set (DagsHub or ALLOW_LOCAL_MLFLOW=true).",
    )
    t1_load = PythonOperator(
        task_id="load_training_data",
        python_callable=_load_training_data,
        doc_md="Load Kaggle CSV (preferred) or yfinance parquet (fallback).",
    )
    t2_preprocess = PythonOperator(
        task_id="run_preprocessing",
        python_callable=_run_preprocessing,
        doc_md="DataPipeline: NaN drop (>30%), median impute, combined feature selection (top 100).",
    )

    # HIGH-18: 4 parallel training tasks (LocalExecutor spawns each as a subprocess)
    t3_lgb = PythonOperator(
        task_id="train_lightgbm",
        python_callable=_train_lightgbm,
        execution_timeout=timedelta(minutes=30),
        doc_md="Train LightGBM from model_params.yaml → lightgbm_final.pkl",
    )
    t3_xgb = PythonOperator(
        task_id="train_xgboost",
        python_callable=_train_xgboost,
        execution_timeout=timedelta(minutes=30),
        doc_md="Train XGBoost from model_params.yaml → xgboost_final.pkl",
    )
    t3_cat = PythonOperator(
        task_id="train_catboost",
        python_callable=_train_catboost,
        execution_timeout=timedelta(minutes=30),
        doc_md="Train CatBoost from model_params.yaml → catboost_final.pkl",
    )
    t3_rf = PythonOperator(
        task_id="train_random_forest",
        python_callable=_train_random_forest,
        execution_timeout=timedelta(minutes=30),
        doc_md="Train RandomForest from model_params.yaml → random_forest_final.pkl",
    )
    t4_ensemble = PythonOperator(
        task_id="train_ensemble",
        python_callable=_train_ensemble,
        execution_timeout=timedelta(minutes=15),
        doc_md="Weighted ensemble from successful sub-models → ensemble_final.pkl (CRIT-5).",
    )
    t5_eval = PythonOperator(
        task_id="evaluate_models",
        python_callable=_evaluate_models,
        doc_md="Compute RMSE/MAE/R²/DirAcc for all models; write metrics.json.",
    )
    t6_promote = PythonOperator(
        task_id="promote_if_better",
        python_callable=_promote_if_better,
        doc_md="Log to MLflow with signature; register and promote best model if RMSE improves.",
    )
    t7_cleanup = PythonOperator(
        task_id="cleanup_splits",
        python_callable=_cleanup_splits,
        trigger_rule="all_done",    # Run even if upstream tasks partially failed
        doc_md="Remove per-run split parquets and stale snapshots (MED-7).",
    )

    # ---------------------------------------------------------------------------
    # Dependency graph
    # ---------------------------------------------------------------------------
    #
    #  t0_validate ──► t1_load ──► t2_preprocess ──► t3_lgb ─┐
    #                                               ──► t3_xgb ──► t4_ensemble ──► t5_eval ──► t6_promote ──► t7_cleanup
    #                                               ──► t3_cat ─┘
    #                                               ──► t3_rf  ─┘
    #
    t0_validate >> t1_load >> t2_preprocess
    t2_preprocess >> [t3_lgb, t3_xgb, t3_cat, t3_rf]
    [t3_lgb, t3_xgb, t3_cat, t3_rf] >> t4_ensemble
    t4_ensemble >> t5_eval >> t6_promote >> t7_cleanup
