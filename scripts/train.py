#!/usr/bin/env python
"""Training script for Hull Tactical models."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import MLflowTrainer
from src.utils.config import Config
from src.utils.logger import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Hull Tactical market prediction models"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["lightgbm", "xgboost", "catboost", "random_forest", "ensemble", "all"],
        help="Model type to train (default: all)",
    )

    parser.add_argument(
        "--n-features",
        type=int,
        default=150,
        help="Number of features to select (default: 150)",
    )

    parser.add_argument(
        "--register",
        action="store_true",
        help="Register model to MLflow registry",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for artifacts",
    )

    parser.add_argument(
        "--cutoff-date",
        type=int,
        default=None,
        help="Cutoff date_id for training data (default: None = use all data)",
    )

    parser.add_argument(
        "--nan-threshold",
        type=float,
        default=0.30,
        help="Drop columns with more than this fraction of NaN values (default: 0.30)",
    )

    parser.add_argument(
        "--use-catboost-imputation",
        action="store_true",
        default=True,
        help="Use CatBoost for imputing missing values (default: True)",
    )

    parser.add_argument(
        "--no-catboost-imputation",
        action="store_true",
        help="Disable CatBoost imputation, use median instead",
    )

    parser.add_argument(
        "--feature-selection-method",
        type=str,
        default="combined",
        choices=["combined", "importance", "correlation", "mutual_info"],
        help="Feature selection method (default: combined)",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run walk-forward validation",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of validation splits (default: 5)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("HULL TACTICAL - MODEL TRAINING")
    logger.info("=" * 80)

    # Determine if CatBoost imputation should be used
    use_catboost = args.use_catboost_imputation and not args.no_catboost_imputation

    # Initialize trainer with preprocessing options
    trainer = MLflowTrainer(
        experiment_name=args.experiment,
        n_features=args.n_features,
        nan_threshold=args.nan_threshold,
        use_catboost_imputation=use_catboost,
        feature_selection_method=args.feature_selection_method,
    )

    # Load and prepare data
    logger.info("\n1. Loading and preparing data...")
    logger.info(f"   - NaN threshold: {args.nan_threshold}")
    logger.info(f"   - CatBoost imputation: {use_catboost}")
    logger.info(f"   - Feature selection: {args.feature_selection_method}")
    logger.info(f"   - Number of features: {args.n_features}")
    logger.info(f"   - Cutoff date: {args.cutoff_date or 'None (all data)'}")

    trainer.load_and_prepare_data(
        cutoff_date_id=args.cutoff_date,
        n_features=args.n_features,
    )

    # Run walk-forward validation if requested
    if args.validate:
        logger.info("\n2. Running walk-forward validation...")

        if args.model != "all":
            results = trainer.run_walk_forward_validation(
                model_type=args.model,
                n_splits=args.n_splits,
            )
            logger.info(f"Validation results: {results['mean_metrics']}")
        else:
            for model_type in ["lightgbm", "xgboost", "catboost", "random_forest"]:
                logger.info(f"\nValidating {model_type}...")
                results = trainer.run_walk_forward_validation(
                    model_type=model_type,
                    n_splits=args.n_splits,
                )
                logger.info(
                    f"  R2: {results['mean_metrics'].get('r2', 0):.4f} "
                    f"(+/- {results['std_metrics'].get('r2', 0):.4f}), "
                    f"Dir.Acc: {results['mean_metrics'].get('directional_accuracy', 0):.2f}%"
                )

    # Train models
    logger.info("\n3. Training models...")

    if args.model == "all":
        # Train all models
        results = trainer.train_all_models(register_best=args.register)

        # Find best model (by R2 score)
        best_model_type = max(
            results.keys(),
            key=lambda k: results[k][1].get("r2", float("-inf")),
        )
        best_model, best_metrics = results[best_model_type]

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 80)

        for model_type, (model, metrics) in results.items():
            logger.info(
                f"\n{model_type.upper()}:"
                f"\n  RMSE: {metrics.get('rmse', 'N/A'):.6f}"
                f"\n  R²: {metrics.get('r2', 'N/A'):.4f}"
                f"\n  Dir.Acc: {metrics.get('directional_accuracy', 'N/A'):.2f}%"
            )

        logger.info(f"\nBest model: {best_model_type.upper()}")

        # Save best model
        trainer.save_artifacts(best_model, args.output_dir)

    elif args.model == "ensemble":
        # Train ensemble
        ensemble, metrics = trainer.train_ensemble()

        logger.info("\n" + "=" * 80)
        logger.info("ENSEMBLE RESULTS")
        logger.info("=" * 80)
        logger.info(f"RMSE: {metrics.get('rmse', 'N/A'):.6f}")
        logger.info(f"R²: {metrics.get('r2', 'N/A'):.4f}")
        logger.info(f"Dir.Acc: {metrics.get('directional_accuracy', 'N/A'):.2f}%")

        trainer.save_artifacts(ensemble, args.output_dir)

    else:
        # Train single model
        model, metrics = trainer.train_model(
            model_type=args.model,
            register_model=args.register,
        )

        logger.info("\n" + "=" * 80)
        logger.info(f"{args.model.upper()} RESULTS")
        logger.info("=" * 80)
        logger.info(f"RMSE: {metrics.get('rmse', 'N/A'):.6f}")
        logger.info(f"R²: {metrics.get('r2', 'N/A'):.4f}")
        logger.info(f"Dir.Acc: {metrics.get('directional_accuracy', 'N/A'):.2f}%")

        trainer.save_artifacts(model, args.output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
