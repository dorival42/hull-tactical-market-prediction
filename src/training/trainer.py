"""Training orchestrator with MLflow integration and advanced preprocessing."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.kaggle_loader import KaggleDataLoader
from src.data.preprocessor import DataPipeline, DataPreprocessor, AdvancedFeatureSelector
from src.features.feature_engineer import FeatureEngineer
from src.models.base import BaseModel, ModelFactory
from src.models.ensemble import EnsembleModel
from src.training.validator import WalkForwardValidator
from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.metrics import ModelMetrics

logger = get_logger(__name__)


class MLflowTrainer:
    """
    Training orchestrator with MLflow integration.

    Features:
    - Advanced data preprocessing with CatBoost imputation
    - No date cutoff (uses all available data)
    - Columns with >30% missing values are dropped
    - Combined feature selection method
    - MLflow experiment tracking
    - Saves all trained models
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        n_features: int = 150,
        nan_threshold: float = 0.30,
        use_catboost_imputation: bool = True,
        feature_selection_method: str = "combined",
    ):
        """
        Initialize trainer.

        Args:
            experiment_name: MLflow experiment name.
            tracking_uri: MLflow tracking URI.
            n_features: Number of features to select.
            nan_threshold: Drop columns with more than this fraction of NaN.
            use_catboost_imputation: Whether to use CatBoost for imputation.
            feature_selection_method: Feature selection method ('combined', 'importance',
                'correlation', 'mutual_info').
        """
        self.config = Config()
        self.experiment_name = experiment_name or self.config.experiment_name
        self.tracking_uri = tracking_uri or self.config.mlflow_tracking_uri
        self.n_features = n_features
        self.nan_threshold = nan_threshold
        self.use_catboost_imputation = use_catboost_imputation
        self.feature_selection_method = feature_selection_method

        self._mlflow = None
        self._setup_mlflow()

        # Initialize components
        self.feature_engineer = FeatureEngineer(verbose=True)
        self.data_pipeline = DataPipeline(
            nan_threshold=nan_threshold,
            use_catboost_imputation=use_catboost_imputation,
            n_features=n_features,
            feature_selection_method=feature_selection_method,
            verbose=True,
        )
        self.validator = WalkForwardValidator()

        # Data storage
        self.train_df: Optional[pd.DataFrame] = None
        self.train_df_processed: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            logger.info(f"MLflow configured: {self.tracking_uri}")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Continuing without tracking.")
            self._mlflow = None

    def load_and_prepare_data(
        self,
        use_feature_engineering: bool = True,
        warmup_period: int = 100,
        cutoff_date_id: Optional[int] = None,
        n_features: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load data and perform complete preprocessing.

        By default, NO DATE CUTOFF - Uses all available data.

        Steps:
        1. Load raw data from Kaggle
        2. Apply feature engineering (optional)
        3. Drop columns with >30% missing values
        4. Impute remaining missing values with CatBoost
        5. Select best features using combined method

        Args:
            use_feature_engineering: Whether to apply feature engineering.
            warmup_period: Rows to skip for warm-up (rolling features).
            cutoff_date_id: Optional date_id cutoff (default: None = use all data).
            n_features: Optional number of features to override instance setting.

        Returns:
            Prepared DataFrame.
        """
        # Override n_features if provided
        if n_features is not None:
            self.n_features = n_features
            self.data_pipeline = DataPipeline(
                nan_threshold=self.nan_threshold,
                use_catboost_imputation=self.use_catboost_imputation,
                n_features=n_features,
                feature_selection_method=self.feature_selection_method,
                verbose=True,
            )
        logger.info("=" * 80)
        logger.info("DATA PREPARATION")
        logger.info("=" * 80)
        if cutoff_date_id is None:
            logger.info("NOTE: Using ALL data - NO date cutoff applied")
        else:
            logger.info(f"NOTE: Using data up to date_id={cutoff_date_id}")

        # Step 1: Load raw data
        logger.info("\n1. Loading data from Kaggle...")
        loader = KaggleDataLoader()
        self.train_df = loader.load_train(cutoff_date_id=cutoff_date_id)
        logger.info(f"   Loaded: {self.train_df.shape[0]} rows x {self.train_df.shape[1]} columns")

        # Step 2: Feature engineering (optional)
        if use_feature_engineering:
            logger.info("\n2. Applying feature engineering...")
            self.train_df = self.feature_engineer.fit_transform(self.train_df)
            logger.info(f"   After FE: {self.train_df.shape[0]} rows x {self.train_df.shape[1]} columns")

        # Step 3: Skip warm-up period (for rolling features)
        if warmup_period > 0:
            logger.info(f"\n3. Skipping warm-up period ({warmup_period} rows)...")
            # Also drop rows with NaN in target
            self.train_df = self.train_df.dropna(subset=["market_forward_excess_returns"])
            self.train_df = self.train_df.iloc[warmup_period:].copy()
            logger.info(f"   After warm-up: {self.train_df.shape[0]} rows")

        # Step 4: Apply data pipeline (preprocessing + feature selection)
        logger.info("\n4. Applying data pipeline...")
        logger.info(f"   - NaN threshold: {self.nan_threshold*100:.0f}%")
        logger.info(f"   - CatBoost imputation: {self.use_catboost_imputation}")
        logger.info(f"   - Feature selection: {self.feature_selection_method}")
        logger.info(f"   - Target features: {self.n_features}")

        # Fit the pipeline on training data
        self.data_pipeline.fit(self.train_df, target_col="market_forward_excess_returns")

        # Get selected features
        self.feature_cols = self.data_pipeline.get_selected_features()

        # Store processed data
        self.train_df_processed = self.data_pipeline.preprocessor.transform(self.train_df)

        logger.info("\n" + "=" * 80)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info(f"Final shape: {self.train_df_processed.shape}")
        logger.info(f"Selected features: {len(self.feature_cols)}")
        logger.info("=" * 80)

        return self.train_df_processed

    def train_model(
        self,
        model_type: str,
        run_name: Optional[str] = None,
        register_model: bool = False,
    ) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Train a single model with MLflow tracking.

        Args:
            model_type: Type of model to train.
            run_name: Optional name for the MLflow run.
            register_model: Whether to register to model registry.

        Returns:
            Tuple of (trained model, metrics dictionary).
        """
        if self.train_df_processed is None:
            self.load_and_prepare_data()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"{model_type}_{timestamp}"

        # Prepare data
        target_col = self.config.target_column
        test_size = self.config.get("training.test_size", 0.2)

        split_idx = int(len(self.train_df_processed) * (1 - test_size))
        train_data = self.train_df_processed.iloc[:split_idx]
        val_data = self.train_df_processed.iloc[split_idx:]

        X_train = train_data[self.feature_cols].fillna(0)
        y_train = train_data[target_col]
        X_val = val_data[self.feature_cols].fillna(0)
        y_val = val_data[target_col]

        logger.info(f"\nTraining {model_type}...")
        logger.info(f"  Train size: {len(X_train)}")
        logger.info(f"  Val size: {len(X_val)}")
        logger.info(f"  Features: {len(self.feature_cols)}")

        # Create model
        model = ModelFactory.create(model_type)

        # Train with MLflow tracking
        if self._mlflow:
            with self._mlflow.start_run(run_name=run_name) as run:
                # Log preprocessing parameters
                self._mlflow.log_param("nan_threshold", self.nan_threshold)
                self._mlflow.log_param("catboost_imputation", self.use_catboost_imputation)
                self._mlflow.log_param("feature_selection_method", self.feature_selection_method)
                self._mlflow.log_param("date_cutoff", "None (all data)")

                # Log model parameters
                self._mlflow.log_params(model.params)
                self._mlflow.log_param("n_features", len(self.feature_cols))
                self._mlflow.log_param("train_size", len(X_train))
                self._mlflow.log_param("val_size", len(X_val))

                # Train model
                model.fit(X_train, y_train, X_val, y_val)

                # Calculate prediction metrics
                y_pred = model.predict(X_val)
                pred_metrics = ModelMetrics.calculate_all_metrics(y_val.values, y_pred)

                # Log metrics
                self._mlflow.log_metrics(model.training_metrics)
                self._mlflow.log_metrics(pred_metrics)

                # Log feature importance
                importance = model.get_feature_importance()
                importance.to_csv("/tmp/feature_importance.csv", index=False)
                self._mlflow.log_artifact("/tmp/feature_importance.csv")

                # Log selected features
                with open("/tmp/selected_features.json", "w") as f:
                    json.dump(self.feature_cols, f)
                self._mlflow.log_artifact("/tmp/selected_features.json")

                # Log model
                self._mlflow.sklearn.log_model(
                    model.model,
                    artifact_path="model",
                    registered_model_name=f"hull-tactical-{model_type}" if register_model else None,
                )

                run_id = run.info.run_id
                logger.info(f"MLflow run completed: {run_id}")
        else:
            model.fit(X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_val)
            pred_metrics = ModelMetrics.calculate_all_metrics(y_val.values, y_pred)

        # Combine metrics
        all_metrics = {**model.training_metrics, **pred_metrics}

        logger.info(f"  RMSE: {all_metrics.get('rmse', 'N/A'):.6f}")
        logger.info(f"  MAE: {all_metrics.get('mae', 'N/A'):.6f}")
        logger.info(f"  R2: {all_metrics.get('r2', 'N/A'):.4f}")
        logger.info(f"  Directional Accuracy: {all_metrics.get('directional_accuracy', 'N/A'):.2f}%")

        return model, all_metrics

    def train_all_models(
        self,
        model_types: Optional[List[str]] = None,
        register_best: bool = True,
        save_all: bool = True,
        output_dir: str = "artifacts",
    ) -> Dict[str, Tuple[BaseModel, Dict[str, float]]]:
        """
        Train all specified model types and save them.

        Args:
            model_types: List of model types to train.
            register_best: Whether to register the best model.
            save_all: Whether to save all trained models.
            output_dir: Output directory for saved models.

        Returns:
            Dictionary mapping model types to (model, metrics) tuples.
        """
        if model_types is None:
            model_types = ["lightgbm", "xgboost", "catboost", "random_forest"]

        results = {}

        for model_type in model_types:
            logger.info(f"\n{'='*40}")
            logger.info(f"Training {model_type.upper()}")
            logger.info(f"{'='*40}")

            try:
                model, metrics = self.train_model(model_type)
                results[model_type] = (model, metrics)

                # Save model
                if save_all:
                    self.save_artifacts(model, output_dir)

            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")

        # Find best model by R2
        if results:
            best_type = max(
                results.keys(),
                key=lambda k: results[k][1].get("r2", float("-inf")),
            )
            logger.info(f"\nBest model (by R2): {best_type}")
            logger.info(f"  R2: {results[best_type][1].get('r2', 'N/A'):.4f}")

            if register_best and self._mlflow:
                self.train_model(best_type, register_model=True)

        return results

    def train_ensemble(
        self,
        model_types: Optional[List[str]] = None,
        run_name: Optional[str] = None,
    ) -> Tuple[EnsembleModel, Dict[str, float]]:
        """
        Train an ensemble model.

        Args:
            model_types: List of model types to include.
            run_name: Optional name for the MLflow run.

        Returns:
            Tuple of (ensemble model, metrics dictionary).
        """
        if self.train_df_processed is None:
            self.load_and_prepare_data()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"ensemble_{timestamp}"

        # Prepare data
        target_col = self.config.target_column
        test_size = self.config.get("training.test_size", 0.2)

        split_idx = int(len(self.train_df_processed) * (1 - test_size))
        train_data = self.train_df_processed.iloc[:split_idx]
        val_data = self.train_df_processed.iloc[split_idx:]

        X_train = train_data[self.feature_cols].fillna(0)
        y_train = train_data[target_col]
        X_val = val_data[self.feature_cols].fillna(0)
        y_val = val_data[target_col]

        # Create ensemble
        ensemble = EnsembleModel.from_config(model_types=model_types)

        logger.info(f"\nTraining Ensemble with {len(ensemble.models)} models...")

        # Train with MLflow tracking
        if self._mlflow:
            with self._mlflow.start_run(run_name=run_name) as run:
                self._mlflow.log_param("ensemble_method", ensemble.method)
                self._mlflow.log_param("n_models", len(ensemble.models))
                self._mlflow.log_param("nan_threshold", self.nan_threshold)
                self._mlflow.log_param("catboost_imputation", self.use_catboost_imputation)
                self._mlflow.log_param("feature_selection_method", self.feature_selection_method)

                # Train ensemble
                ensemble.fit(X_train, y_train, X_val, y_val)

                # Calculate prediction metrics
                y_pred = ensemble.predict(X_val)
                pred_metrics = ModelMetrics.calculate_all_metrics(y_val.values, y_pred)

                # Log metrics
                self._mlflow.log_metrics(ensemble.training_metrics)
                self._mlflow.log_metrics(pred_metrics)

                logger.info(f"Ensemble MLflow run completed: {run.info.run_id}")
        else:
            ensemble.fit(X_train, y_train, X_val, y_val)
            y_pred = ensemble.predict(X_val)
            pred_metrics = ModelMetrics.calculate_all_metrics(y_val.values, y_pred)

        all_metrics = {**ensemble.training_metrics, **pred_metrics}

        logger.info(f"  RMSE: {all_metrics.get('rmse', 'N/A'):.6f}")
        logger.info(f"  R2: {all_metrics.get('r2', 'N/A'):.4f}")
        logger.info(f"  Directional Accuracy: {all_metrics.get('directional_accuracy', 'N/A'):.2f}%")

        return ensemble, all_metrics

    def run_walk_forward_validation(
        self,
        model_type: str,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            model_type: Type of model to validate.
            n_splits: Number of validation splits.

        Returns:
            Validation results dictionary.
        """
        if self.train_df_processed is None:
            self.load_and_prepare_data()

        target_col = self.config.target_column

        logger.info(f"\nRunning walk-forward validation for {model_type}...")

        results = self.validator.validate(
            df=self.train_df_processed,
            feature_cols=self.feature_cols,
            target_col=target_col,
            model_factory=lambda: ModelFactory.create(model_type),
            n_splits=n_splits,
        )

        # Log to MLflow if available
        if self._mlflow:
            with self._mlflow.start_run(run_name=f"wf_validation_{model_type}"):
                self._mlflow.log_param("model_type", model_type)
                self._mlflow.log_param("n_splits", n_splits)
                self._mlflow.log_param("nan_threshold", self.nan_threshold)
                self._mlflow.log_param("catboost_imputation", self.use_catboost_imputation)
                self._mlflow.log_param("feature_selection_method", self.feature_selection_method)

                for metric_name, values in results["fold_metrics"].items():
                    self._mlflow.log_metric(f"avg_{metric_name}", np.mean(values))
                    self._mlflow.log_metric(f"std_{metric_name}", np.std(values))

        return results

    def save_artifacts(
        self,
        model: BaseModel,
        output_dir: str = "artifacts",
    ) -> None:
        """
        Save model and feature list to local artifacts.

        Args:
            model: Trained model to save.
            output_dir: Output directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_path / f"{model.name.lower()}_final.pkl"
        model.save(str(model_path))
        logger.info(f"Model saved: {model_path}")

        # Save feature list
        self.data_pipeline.save(str(output_path))

        # Save preprocessing info
        info = {
            "nan_threshold": self.nan_threshold,
            "catboost_imputation": self.use_catboost_imputation,
            "feature_selection_method": self.feature_selection_method,
            "n_features": len(self.feature_cols),
            "total_train_samples": len(self.train_df_processed) if self.train_df_processed is not None else 0,
            "date_cutoff": "None (all data used)",
        }

        with open(output_path / "preprocessing_info.json", "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"Artifacts saved to {output_path}")

    def get_preprocessing_report(self) -> pd.DataFrame:
        """
        Get a report of preprocessing statistics.

        Returns:
            DataFrame with preprocessing report.
        """
        if self.train_df is None:
            raise ValueError("No data loaded. Call load_and_prepare_data() first.")

        return self.data_pipeline.preprocessor.get_preprocessing_report(self.train_df)
