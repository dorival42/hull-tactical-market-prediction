#!/usr/bin/env python
"""Submission script for Kaggle competition."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.data.kaggle_loader import KaggleDataLoader
from src.features.feature_engineer import FeatureEngineer
from src.inference.predictor import Predictor
from src.utils.logger import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Kaggle submission with predictions"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model file",
    )

    parser.add_argument(
        "--features-path",
        type=str,
        default="artifacts/selected_features.json",
        help="Path to features JSON file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="submission.parquet",
        help="Output file path",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate submission format",
    )

    return parser.parse_args()


def validate_submission(predictions: np.ndarray) -> bool:
    """Validate submission predictions."""
    # Check for NaN
    if np.isnan(predictions).any():
        return False

    # Check for inf
    if np.isinf(predictions).any():
        return False

    return True


def main():
    """Main submission function."""
    args = parse_args()

    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("HULL TACTICAL - SUBMISSION GENERATOR")
    logger.info("=" * 80)

    # Load test data
    logger.info("\n1. Loading test data...")
    loader = KaggleDataLoader()
    test_df = loader.load_test()
    logger.info(f"Test data shape: {test_df.shape}")

    # Load model
    logger.info("\n2. Loading model...")
    predictor = Predictor(model_path=args.model_path)

    if Path(args.features_path).exists():
        predictor.load_features(args.features_path)

    # Feature engineering
    logger.info("\n3. Applying feature engineering...")
    fe = FeatureEngineer(verbose=False)
    test_transformed = fe.transform(test_df)

    # Make predictions
    logger.info("\n4. Generating predictions...")
    result = predictor.predict(test_transformed)

    predictions = np.array(result["predictions"])
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"  Mean prediction: {predictions.mean():.6f}")
    logger.info(f"  Std prediction: {predictions.std():.6f}")
    logger.info(f"  Min prediction: {predictions.min():.6f}")
    logger.info(f"  Max prediction: {predictions.max():.6f}")

    # Validate
    if args.validate:
        logger.info("\n5. Validating submission...")
        if validate_submission(predictions):
            logger.info("  Validation PASSED")
        else:
            logger.error("  Validation FAILED")
            sys.exit(1)

    # Create submission
    logger.info("\n6. Creating submission file...")

    submission = pd.DataFrame({
        "date_id": test_df["date_id"],
        "prediction": predictions,
    })

    # Save as parquet
    output_path = Path(args.output)
    submission.to_parquet(output_path, index=False)
    logger.info(f"Submission saved to: {output_path}")

    # Also save CSV for inspection
    csv_path = output_path.with_suffix(".csv")
    submission.to_csv(csv_path, index=False)
    logger.info(f"CSV copy saved to: {csv_path}")

    logger.info("\n" + "=" * 80)
    logger.info("SUBMISSION COMPLETE")
    logger.info("=" * 80)

    # Print summary
    print("\nSubmission Summary:")
    print(f"  Total rows: {len(submission)}")
    print(f"  Mean prediction: {predictions.mean():.6f}")
    print(f"  Std prediction: {predictions.std():.6f}")


if __name__ == "__main__":
    main()
