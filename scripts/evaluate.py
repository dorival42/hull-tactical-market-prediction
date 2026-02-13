#!/usr/bin/env python
"""Evaluation script for Hull Tactical models."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data.kaggle_loader import KaggleDataLoader
from src.features.feature_engineer import FeatureEngineer
from src.inference.predictor import Predictor
from src.utils.logger import setup_logging, get_logger
from src.utils.metrics import ModelMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Hull Tactical models"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model file",
    )

    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/models",
        help="Directory containing model artifacts",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size fraction (default: 0.2)",
    )

    return parser.parse_args()


def find_models(artifacts_dir: Path) -> list:
    """Find all model files in artifacts directory."""
    return list(artifacts_dir.glob("*_final.pkl"))


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("HULL TACTICAL - MODEL EVALUATION")
    logger.info("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("\n1. Loading data...")
    loader = KaggleDataLoader()
    train_df = loader.load_train(cutoff_date_id=None)  # Use all data

    # Feature engineering
    logger.info("\n2. Feature engineering...")
    fe = FeatureEngineer(verbose=False)
    train_df = fe.fit_transform(train_df)
    train_df = train_df.iloc[100:].copy()  # Skip warm-up

    # Split data
    split_idx = int(len(train_df) * (1 - args.test_size))
    test_df = train_df.iloc[split_idx:]

    logger.info(f"Test set size: {len(test_df)} rows")

    # Load features
    artifacts_dir = Path(args.artifacts_dir)
    features_path = artifacts_dir / "selected_features.json"

    if features_path.exists():
        with open(features_path, "r") as f:
            feature_cols = json.load(f)
        logger.info(f"Loaded {len(feature_cols)} features")
    else:
        logger.warning("No feature file found, using all numeric columns")
        feature_cols = test_df.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()

    # Find models to evaluate
    if args.model_path:
        model_paths = [Path(args.model_path)]
    else:
        model_paths = find_models(artifacts_dir)

    if not model_paths:
        logger.error("No models found to evaluate")
        sys.exit(1)

    logger.info(f"\n3. Evaluating {len(model_paths)} models...")

    # Prepare test data
    target_col = "market_forward_excess_returns"
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[target_col].values

    # Evaluate each model
    results = []

    for model_path in model_paths:
        logger.info(f"\nEvaluating: {model_path.name}")

        try:
            predictor = Predictor(model_path=str(model_path))
            predictor.feature_cols = feature_cols

            # Get predictions
            y_pred = predictor.model.predict(X_test)

            # Calculate metrics
            metrics = ModelMetrics.calculate_all_metrics(y_test, y_pred)

            result = {
                "model": model_path.stem,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "mape": metrics["mape"],
                "directional_accuracy": metrics["directional_accuracy"],
            }

            results.append(result)

            logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            logger.info(f"  MAE: {metrics['mae']:.6f}")
            logger.info(f"  R2: {metrics['r2']:.4f}")
            logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")

        except Exception as e:
            logger.error(f"  Failed: {e}")

    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("r2", ascending=False)

    # Save results
    results_path = output_dir / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)

    print(results_df.to_string(index=False))

    # Best model
    best = results_df.iloc[0]
    logger.info(f"\nBest Model: {best['model']}")
    logger.info(f"  R2: {best['r2']:.4f}")
    logger.info(f"  RMSE: {best['rmse']:.6f}")
    logger.info(f"  Directional Accuracy: {best['directional_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
