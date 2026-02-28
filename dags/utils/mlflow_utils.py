"""
MLflow helpers shared across DAGs.
Provides safe wrappers that log a warning (instead of raising) when the
MLflow server is temporarily unavailable, so DAG tasks remain idempotent.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "hull-tactical-airflow"


def get_mlflow_client():
    """Return an MlflowClient pointed at MLFLOW_URI."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_URI)
    return MlflowClient()


def get_or_create_experiment(name: str = EXPERIMENT_NAME) -> str:
    """
    Return the experiment ID, creating the experiment if necessary.

    Args:
        name: MLflow experiment name.

    Returns:
        Experiment ID string.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        logger.info("Created MLflow experiment '%s' (id=%s)", name, exp_id)
    else:
        exp_id = exp.experiment_id
    return exp_id


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------
def log_run(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    tags: Optional[Dict[str, str]] = None,
    artifact_paths: Optional[list[str]] = None,
    run_name: str = "airflow-run",
) -> Optional[str]:
    """
    Start an MLflow run, log params/metrics/artifacts, and return the run_id.
    Returns None on failure so the caller can decide whether to raise.

    Args:
        params: Hyperparameters / config dict.
        metrics: Evaluation metrics dict.
        tags: Optional run tags.
        artifact_paths: List of local file paths to upload as artifacts.
        run_name: Human-readable run name shown in the MLflow UI.

    Returns:
        run_id string, or None if logging failed.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        exp_id = get_or_create_experiment()

        with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if artifact_paths:
                for path in artifact_paths:
                    if Path(path).exists():
                        mlflow.log_artifact(path)
                    else:
                        logger.warning("Artifact not found, skipping: %s", path)
            run_id = run.info.run_id
            logger.info("MLflow run logged: %s (id=%s)", run_name, run_id)
            return run_id
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): %s", exc)
        return None


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------
def register_model_if_better(
    run_id: str,
    model_artifact_path: str,
    model_name: str,
    metric_name: str,
    new_metric_value: float,
    lower_is_better: bool = True,
) -> bool:
    """
    Register a model version in the MLflow Model Registry and transition it
    to 'Production' only if it outperforms the current production model.

    Args:
        run_id: ID of the MLflow run that trained this model.
        model_artifact_path: Relative artifact path within the run (e.g. 'model').
        model_name: Registered model name in the registry.
        metric_name: Metric used for comparison (e.g. 'rmse').
        new_metric_value: Value of that metric for the new model.
        lower_is_better: True for loss metrics (RMSE, MAE), False for score metrics (R²).

    Returns:
        True if the new model was promoted to Production, False otherwise.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        # --- Register new version ---
        model_uri = f"runs:/{run_id}/{model_artifact_path}"
        new_version = mlflow.register_model(model_uri, model_name)
        logger.info("Registered %s v%s", model_name, new_version.version)

        # --- Get current production metric ---
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        should_promote = True

        if prod_versions:
            prod_run = client.get_run(prod_versions[0].run_id)
            prod_metric = prod_run.data.metrics.get(metric_name)
            if prod_metric is not None:
                if lower_is_better:
                    should_promote = new_metric_value < prod_metric
                else:
                    should_promote = new_metric_value > prod_metric
                logger.info(
                    "Comparison: new=%s=%.6f vs prod=%.6f → promote=%s",
                    metric_name, new_metric_value, prod_metric, should_promote,
                )

        if should_promote:
            # Archive old production model first
            for v in prod_versions:
                client.transition_model_version_stage(
                    model_name, v.version, stage="Archived"
                )
            client.transition_model_version_stage(
                model_name, new_version.version, stage="Production"
            )
            logger.info("✅ %s v%s promoted to Production", model_name, new_version.version)
        else:
            client.transition_model_version_stage(
                model_name, new_version.version, stage="Archived"
            )
            logger.info("New model not better than production — archived.")

        return should_promote

    except Exception as exc:
        logger.warning("Model registry step failed (non-fatal): %s", exc)
        return False
