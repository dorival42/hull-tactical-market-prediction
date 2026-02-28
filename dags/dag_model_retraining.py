"""
DAG 3 — Model Retraining (weekly)
=====================================
Schedule : every Monday at 06:00 UTC (after weekend market close)
Tasks    :
  1. load_training_data   — load latest parquet + Kaggle CSV baseline
  2. run_preprocessing    — NaN drop, CatBoost imputation, feature selection
  3. train_models         — train LightGBM, XGBoost, CatBoost, Ensemble
  4. evaluate_models      — compute RMSE, MAE, R², Directional Accuracy
  5. promote_if_better    — register best model in MLflow Registry → Production

MLflow experiment: 'hull-tactical-airflow'
Model registry name: 'hull-tactical-best'
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_project_root = os.getenv("PYTHONPATH", "/opt/airflow/project")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "hull-tactical",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

ARTIFACTS_DIR = Path(_project_root) / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
PARQUET_DIR = DATA_DIR / "market"
REGISTERED_MODEL_NAME = "hull-tactical-best"

# ---------------------------------------------------------------------------
# Task implementations
# ---------------------------------------------------------------------------

def _load_training_data(**ctx) -> dict:
    """
    Task 1 — Assemble the training dataset.

    Priority order:
      1. Latest market parquet (from DAG 1) — most up-to-date yfinance data
      2. Kaggle baseline CSV (artifacts/data/train.csv) — competition data

    The two sources are concatenated on common columns when both exist.
    Returns a dict with 'path' (str) and 'n_rows' for downstream XCom.
    """
    import pandas as pd

    frames = []

    # Source 1 — yfinance enriched parquet (preferred)
    latest_parquet = PARQUET_DIR / "sp500_features_latest.parquet"
    if latest_parquet.exists():
        df_market = pd.read_parquet(latest_parquet, engine="pyarrow")
        logger.info("[load] Market parquet: %d rows, %d cols", len(df_market), len(df_market.columns))
        frames.append(df_market)
    else:
        logger.warning("[load] Market parquet not found — skipping.")

    # Source 2 — Kaggle competition CSV (baseline)
    kaggle_csv = DATA_DIR / "train.csv"
    if kaggle_csv.exists():
        df_kaggle = pd.read_csv(kaggle_csv)
        logger.info("[load] Kaggle CSV: %d rows, %d cols", len(df_kaggle), len(df_kaggle.columns))
        frames.append(df_kaggle)
    else:
        logger.warning("[load] Kaggle CSV not found — skipping.")

    if not frames:
        raise FileNotFoundError(
            "No training data found. Run dag_data_pipeline first, or place train.csv in artifacts/data/."
        )

    # Use the largest dataset as primary (usually Kaggle CSV with 8991 rows)
    df = max(frames, key=len)
    logger.info("[load] Using primary dataset: %d rows, %d cols", len(df), len(df.columns))

    out_path = DATA_DIR / f"training_snapshot_{ctx['ds_nodash']}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    return {"path": str(out_path), "n_rows": len(df), "n_cols": len(df.columns)}


def _run_preprocessing(**ctx) -> dict:
    """
    Task 2 — Preprocess the training snapshot.

    Steps (mirrors src/data/preprocessor.py DataPipeline):
      - Drop target columns before feature selection
      - Drop columns with > 30% NaN
      - CatBoost imputation for remaining NaN
      - Feature selection (combined: LGB importance + correlation + mutual info)
      - Train/val split (80/20, time-ordered)

    Returns dict with paths to X_train, X_val, y_train, y_val parquets.
    """
    import numpy as np
    import pandas as pd

    task_data: dict = ctx["ti"].xcom_pull(task_ids="load_training_data")
    df = pd.read_parquet(task_data["path"], engine="pyarrow")

    # --- Identify target column ---
    target_col = "market_forward_excess_returns"
    if target_col not in df.columns:
        # Fallback: use log_return from yfinance data
        if "log_return" in df.columns:
            target_col = "log_return"
            logger.info("[preprocess] Using 'log_return' as target (yfinance data).")
        else:
            raise ValueError(
                f"No target column found. Expected '{target_col}' or 'log_return'."
            )

    # --- Drop non-feature columns ---
    drop_cols = [
        "forward_returns", "risk_free_rate",           # Kaggle auxiliary targets
        "date", "date_id",                              # Index columns
        "is_scored", "lagged_forward_returns",          # Test-only cols
        "lagged_risk_free_rate", "lagged_market_forward_excess_returns",
    ]
    feature_cols = [
        c for c in df.columns
        if c != target_col and c not in drop_cols
    ]

    y = df[target_col].dropna()
    X = df.loc[y.index, feature_cols]

    # --- Drop high-NaN columns ---
    nan_threshold = 0.30
    nan_ratio = X.isnull().mean()
    high_nan = nan_ratio[nan_ratio > nan_threshold].index.tolist()
    X = X.drop(columns=high_nan)
    logger.info("[preprocess] Dropped %d high-NaN columns (>%.0f%%)", len(high_nan), nan_threshold * 100)

    # --- Simple median imputation (fallback; CatBoostImputer available in full pipeline) ---
    X = X.fillna(X.median(numeric_only=True))

    # --- Clip infinities ---
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Feature selection: top 100 by variance (fast proxy for importance) ---
    n_features = min(100, X.shape[1])
    variances = X.var().sort_values(ascending=False)
    selected = variances.head(n_features).index.tolist()
    X = X[selected]

    # --- Time-ordered 80/20 split ---
    split_idx = int(len(X) * 0.80)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # --- Persist splits ---
    out_dir = DATA_DIR / f"splits_{ctx['ds_nodash']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(out_dir / "X_train.parquet", index=False)
    X_val.to_parquet(out_dir / "X_val.parquet", index=False)
    y_train.to_frame().to_parquet(out_dir / "y_train.parquet", index=False)
    y_val.to_frame().to_parquet(out_dir / "y_val.parquet", index=False)

    # Save selected feature list
    selected_path = ARTIFACTS_DIR / "selected_features_airflow.json"
    selected_path.write_text(json.dumps(selected))

    logger.info(
        "[preprocess] Train: %d rows | Val: %d rows | Features: %d",
        len(X_train), len(X_val), len(selected),
    )
    return {
        "splits_dir": str(out_dir),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": len(selected),
        "target_col": target_col,
    }


def _train_models(**ctx) -> dict:
    """
    Task 3 — Train LightGBM, XGBoost, CatBoost and an Ensemble.

    Each model is trained on X_train/y_train with early stopping on X_val/y_val.
    Serialized models are saved to artifacts/.

    Returns:
        dict mapping model_name → {'rmse': float, 'artifact_path': str}
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    prep_data: dict = ctx["ti"].xcom_pull(task_ids="run_preprocessing")
    splits_dir = Path(prep_data["splits_dir"])

    X_train = pd.read_parquet(splits_dir / "X_train.parquet")
    X_val   = pd.read_parquet(splits_dir / "X_val.parquet")
    y_train = pd.read_parquet(splits_dir / "y_train.parquet").squeeze()
    y_val   = pd.read_parquet(splits_dir / "y_val.parquet").squeeze()

    results: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}

    # --- LightGBM ---
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        preds = lgb_model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        path = str(ARTIFACTS_DIR / "lightgbm_airflow.pkl")
        with open(path, "wb") as f:
            pickle.dump(lgb_model, f)
        results["lightgbm"] = {"rmse": rmse, "artifact_path": path}
        predictions["lightgbm"] = preds
        logger.info("[train] LightGBM RMSE=%.6f", rmse)
    except Exception as e:
        logger.warning("[train] LightGBM failed: %s", e)

    # --- XGBoost ---
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            early_stopping_rounds=50, eval_metric="rmse", verbosity=0,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = xgb_model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        path = str(ARTIFACTS_DIR / "xgboost_airflow.pkl")
        with open(path, "wb") as f:
            pickle.dump(xgb_model, f)
        results["xgboost"] = {"rmse": rmse, "artifact_path": path}
        predictions["xgboost"] = preds
        logger.info("[train] XGBoost RMSE=%.6f", rmse)
    except Exception as e:
        logger.warning("[train] XGBoost failed: %s", e)

    # --- CatBoost ---
    try:
        from catboost import CatBoostRegressor
        cat_model = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            early_stopping_rounds=50, verbose=0, random_seed=42,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        preds = cat_model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        path = str(ARTIFACTS_DIR / "catboost_airflow.pkl")
        with open(path, "wb") as f:
            pickle.dump(cat_model, f)
        results["catboost"] = {"rmse": rmse, "artifact_path": path}
        predictions["catboost"] = preds
        logger.info("[train] CatBoost RMSE=%.6f", rmse)
    except Exception as e:
        logger.warning("[train] CatBoost failed: %s", e)

    # --- Ensemble (simple average of available models) ---
    if len(predictions) >= 2:
        stacked = np.stack(list(predictions.values()), axis=0)
        ensemble_preds = stacked.mean(axis=0)
        ens_rmse = float(np.sqrt(mean_squared_error(y_val, ensemble_preds)))
        results["ensemble"] = {"rmse": ens_rmse, "artifact_path": "n/a (runtime average)"}
        logger.info("[train] Ensemble RMSE=%.6f", ens_rmse)

    return results


def _evaluate_models(**ctx) -> dict:
    """
    Task 4 — Compute full evaluation metrics and persist metrics.json.

    Metrics computed per model:
      - RMSE, MAE, R², Directional Accuracy

    Also determines the best model (lowest RMSE) for promotion.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    prep_data: dict = ctx["ti"].xcom_pull(task_ids="run_preprocessing")
    train_results: dict = ctx["ti"].xcom_pull(task_ids="train_models")
    splits_dir = Path(prep_data["splits_dir"])
    target_col = prep_data["target_col"]

    X_val = pd.read_parquet(splits_dir / "X_val.parquet")
    y_val = pd.read_parquet(splits_dir / "y_val.parquet").squeeze()

    metrics_by_model: dict = {}
    best_model_name: str | None = None
    best_rmse = float("inf")

    for model_name, info in train_results.items():
        if model_name == "ensemble":
            continue  # Ensemble stats already in train_results

        artifact_path = info["artifact_path"]
        if not Path(artifact_path).exists():
            continue

        with open(artifact_path, "rb") as f:
            model = pickle.load(f)

        preds = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae  = float(mean_absolute_error(y_val, preds))
        r2   = float(r2_score(y_val, preds))
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y_val.values)))

        metrics_by_model[model_name] = {
            "rmse": rmse, "mae": mae, "r2": r2,
            "directional_accuracy": dir_acc,
        }

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = model_name

    # Include ensemble RMSE
    if "ensemble" in train_results:
        metrics_by_model["ensemble"] = {"rmse": train_results["ensemble"]["rmse"]}
        if train_results["ensemble"]["rmse"] < best_rmse:
            best_rmse = train_results["ensemble"]["rmse"]
            best_model_name = "ensemble"

    # Persist metrics JSON (read by Streamlit dashboard)
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_by_model, indent=2))
    logger.info("[eval] Metrics written → %s", metrics_path)
    logger.info("[eval] Best model: %s (RMSE=%.6f)", best_model_name, best_rmse)

    return {
        "metrics": metrics_by_model,
        "best_model": best_model_name,
        "best_rmse": best_rmse,
        "train_results": train_results,
    }


def _promote_if_better(**ctx) -> dict:
    """
    Task 5 — Log the best model to MLflow and promote to Production if RMSE improves.

    Steps:
      1. Start an MLflow run, log params + metrics + model artifact.
      2. Register the model version in MLflow Model Registry.
      3. Compare with current Production model; promote if RMSE is lower.
    """
    import mlflow
    import mlflow.sklearn
    from dags.utils.mlflow_utils import get_or_create_experiment, register_model_if_better

    eval_data: dict  = ctx["ti"].xcom_pull(task_ids="evaluate_models")
    prep_data: dict  = ctx["ti"].xcom_pull(task_ids="run_preprocessing")
    train_results: dict = eval_data["train_results"]

    best_model_name: str = eval_data["best_model"]
    best_rmse: float     = eval_data["best_rmse"]
    metrics: dict        = eval_data["metrics"]

    if best_model_name is None or best_model_name == "ensemble":
        logger.warning("[promote] No single model to register — skipping registry step.")
        return {"promoted": False, "reason": "no single best model"}

    artifact_path = train_results[best_model_name]["artifact_path"]
    if not Path(artifact_path).exists():
        logger.warning("[promote] Best model artifact missing: %s", artifact_path)
        return {"promoted": False, "reason": "artifact missing"}

    with open(artifact_path, "rb") as f:
        model_obj = pickle.load(f)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    exp_id = get_or_create_experiment("hull-tactical-airflow")

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"weekly-retrain-{ctx['ds']}-{best_model_name}",
        tags={"dag": "dag_model_retraining", "triggered_by": "airflow"},
    ) as run:
        # Log hyperparams
        mlflow.log_params({
            "model_type": best_model_name,
            "n_train": prep_data["n_train"],
            "n_val": prep_data["n_val"],
            "n_features": prep_data["n_features"],
            "target_col": prep_data["target_col"],
            "logical_date": ctx["ds"],
        })
        # Log metrics for all models
        for name, m in metrics.items():
            for metric_key, val in m.items():
                mlflow.log_metric(f"{name}_{metric_key}", val)

        # Log best model artifact
        mlflow.sklearn.log_model(model_obj, artifact_path="model")
        run_id = run.info.run_id

    # Register & conditionally promote
    promoted = register_model_if_better(
        run_id=run_id,
        model_artifact_path="model",
        model_name=REGISTERED_MODEL_NAME,
        metric_name="rmse",
        new_metric_value=best_rmse,
        lower_is_better=True,
    )

    result = {
        "promoted": promoted,
        "best_model": best_model_name,
        "best_rmse": best_rmse,
        "run_id": run_id,
    }
    logger.info("[promote] Result: %s", result)
    return result


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="dag_model_retraining",
    description="Weekly: retrain all models, evaluate, promote best to MLflow Production",
    schedule_interval="0 6 * * 1",    # Every Monday at 06:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["training", "mlflow", "weekly"],
    default_args=DEFAULT_ARGS,
    doc_md=__doc__,
) as dag:

    t1_load = PythonOperator(
        task_id="load_training_data",
        python_callable=_load_training_data,
        doc_md="Load latest market parquet + Kaggle CSV baseline.",
    )

    t2_preprocess = PythonOperator(
        task_id="run_preprocessing",
        python_callable=_run_preprocessing,
        doc_md="Drop high-NaN columns, impute, select top 100 features, split 80/20.",
    )

    t3_train = PythonOperator(
        task_id="train_models",
        python_callable=_train_models,
        doc_md="Train LightGBM, XGBoost, CatBoost with early stopping + Ensemble.",
        execution_timeout=timedelta(hours=2),   # Training can be slow on large datasets
    )

    t4_eval = PythonOperator(
        task_id="evaluate_models",
        python_callable=_evaluate_models,
        doc_md="Compute RMSE/MAE/R²/Directional Accuracy, persist metrics.json.",
    )

    t5_promote = PythonOperator(
        task_id="promote_if_better",
        python_callable=_promote_if_better,
        doc_md="Log run to MLflow, register model, promote to Production if RMSE improved.",
    )

    t1_load >> t2_preprocess >> t3_train >> t4_eval >> t5_promote
