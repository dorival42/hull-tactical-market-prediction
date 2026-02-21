#!/usr/bin/env python
"""Quick test script to verify MLflow tracking on DagsHub."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # Setup MLflow with DagsHub
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/dorival42/hull-tactical-market-prediction.mlflow/",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("hull-tactical-experiments")

    print(f"MLflow Tracking URI: {tracking_uri}")
    print("Generating synthetic market data...")

    # Generate synthetic data mimicking market features
    X, y = make_regression(
        n_samples=5000,
        n_features=50,
        n_informative=20,
        noise=0.1,
        random_state=42,
    )

    # Rename to simulate market features
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Train models and log to MLflow
    models_config = [
        {
            "name": "GradientBoosting-v1",
            "params": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
            },
        },
        {
            "name": "GradientBoosting-v2",
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
            },
        },
        {
            "name": "GradientBoosting-v3",
            "params": {
                "n_estimators": 300,
                "max_depth": 3,
                "learning_rate": 0.01,
                "subsample": 0.7,
            },
        },
    ]

    best_rmse = float("inf")
    best_run = None

    for config in models_config:
        run_name = config["name"]
        params = config["params"]

        print(f"\n{'='*60}")
        print(f"Training: {run_name}")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("n_features", 50)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("feature_selection_method", "combined")
            mlflow.log_param("nan_threshold", 0.30)

            # Train model
            model = GradientBoostingRegressor(**params, random_state=42)
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            # Directional accuracy (simulated)
            dir_acc = np.mean(np.sign(y_pred_test) == np.sign(y_test)) * 100

            # Log metrics
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("directional_accuracy", dir_acc)
            mlflow.log_metric("mae", float(np.mean(np.abs(y_test - y_pred_test))))

            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"hull-tactical-test-{run_name.lower()}",
            )

            # Log feature importance as artifact
            importance = model.feature_importances_
            top_features = sorted(
                zip(feature_names, importance), key=lambda x: x[1], reverse=True
            )[:10]

            print(f"  Train RMSE: {train_rmse:.4f}")
            print(f"  Test  RMSE: {test_rmse:.4f}")
            print(f"  Train R²:   {train_r2:.4f}")
            print(f"  Test  R²:   {test_r2:.4f}")
            print(f"  Dir. Acc:   {dir_acc:.1f}%")
            print(f"  Top features: {[f[0] for f in top_features[:5]]}")

            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_run = run_name

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_run} (RMSE: {best_rmse:.4f})")
    print(f"{'='*60}")
    print(f"\nView results on DagsHub:")
    print(f"  {tracking_uri}")
    print("\nDone! Check your DagsHub dashboard for the logged experiments.")


if __name__ == "__main__":
    main()
