"""Walk-forward validation for Hull Tactical."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.logger import get_logger
from src.utils.metrics import ModelMetrics

logger = get_logger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.

    Implements expanding window validation where each fold
    uses all previous data for training.
    """

    def __init__(self):
        """Initialize validator."""
        self.results: Dict[str, Any] = {}

    def validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model_factory: Callable[[], BaseModel],
        n_splits: int = 5,
        date_column: str = "date_id",
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            df: Full DataFrame.
            feature_cols: Feature column names.
            target_col: Target column name.
            model_factory: Callable that returns a new model instance.
            n_splits: Number of validation splits.
            date_column: Column for temporal ordering.

        Returns:
            Dictionary with validation results.
        """
        logger.info(f"Starting walk-forward validation with {n_splits} splits")

        df = df.sort_values(date_column).reset_index(drop=True)
        total_size = len(df)
        test_size = total_size // (n_splits + 1)

        fold_results = []
        all_predictions = []
        all_actuals = []

        for fold in range(n_splits):
            logger.info(f"Fold {fold + 1}/{n_splits}")

            # Calculate split indices
            train_end = (fold + 1) * test_size + (total_size - (n_splits + 1) * test_size)
            test_start = train_end
            test_end = min(train_end + test_size, total_size)

            train_fold = df.iloc[:train_end]
            test_fold = df.iloc[test_start:test_end]

            logger.info(f"  Train: {len(train_fold)} rows, Test: {len(test_fold)} rows")

            # Prepare data
            X_train = train_fold[feature_cols].fillna(0)
            y_train = train_fold[target_col]
            X_test = test_fold[feature_cols].fillna(0)
            y_test = test_fold[target_col]

            # Train and predict
            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            fold_metrics = ModelMetrics.calculate_all_metrics(y_test.values, y_pred)
            fold_results.append(fold_metrics)

            all_predictions.extend(y_pred)
            all_actuals.extend(y_test.values)

            logger.info(
                f"  RMSE: {fold_metrics['rmse']:.6f}, "
                f"R2: {fold_metrics['r2']:.4f}, "
                f"Dir.Acc: {fold_metrics['directional_accuracy']:.2f}%"
            )

        # Aggregate results
        self.results = self._aggregate_results(
            fold_results, all_predictions, all_actuals
        )

        return self.results

    def _aggregate_results(
        self,
        fold_results: List[Dict[str, float]],
        all_predictions: List[float],
        all_actuals: List[float],
    ) -> Dict[str, Any]:
        """Aggregate results from all folds."""
        # Per-metric aggregation
        fold_metrics: Dict[str, List[float]] = {}
        for result in fold_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in fold_metrics:
                        fold_metrics[key] = []
                    fold_metrics[key].append(value)

        # Global metrics
        global_metrics = ModelMetrics.calculate_all_metrics(
            np.array(all_actuals), np.array(all_predictions)
        )

        return {
            "n_splits": len(fold_results),
            "fold_metrics": fold_metrics,
            "mean_metrics": {k: np.mean(v) for k, v in fold_metrics.items()},
            "std_metrics": {k: np.std(v) for k, v in fold_metrics.items()},
            "global_metrics": global_metrics,
            "all_predictions": all_predictions,
            "all_actuals": all_actuals,
        }

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of validation results.

        Returns:
            DataFrame with summary statistics.
        """
        if not self.results:
            raise ValueError("No validation results. Run validate() first.")

        summary_data = []
        for metric, values in self.results["fold_metrics"].items():
            summary_data.append(
                {
                    "metric": metric,
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "global": self.results["global_metrics"].get(metric),
                }
            )

        return pd.DataFrame(summary_data)

    def compare_models(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model_factories: Dict[str, Callable[[], BaseModel]],
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """
        Compare multiple models using walk-forward validation.

        Args:
            df: Full DataFrame.
            feature_cols: Feature column names.
            target_col: Target column name.
            model_factories: Dictionary mapping names to model factories.
            n_splits: Number of validation splits.

        Returns:
            DataFrame comparing model performance.
        """
        comparison_results = []

        for model_name, factory in model_factories.items():
            logger.info(f"Validating {model_name}...")

            results = self.validate(
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                model_factory=factory,
                n_splits=n_splits,
            )

            comparison_results.append(
                {
                    "model": model_name,
                    "rmse_mean": results["mean_metrics"].get("rmse"),
                    "rmse_std": results["std_metrics"].get("rmse"),
                    "r2_mean": results["mean_metrics"].get("r2"),
                    "r2_std": results["std_metrics"].get("r2"),
                    "dir_acc_mean": results["mean_metrics"].get("directional_accuracy"),
                    "dir_acc_std": results["std_metrics"].get("directional_accuracy"),
                    "global_rmse": results["global_metrics"].get("rmse"),
                    "global_r2": results["global_metrics"].get("r2"),
                }
            )

        return pd.DataFrame(comparison_results).sort_values(
            "r2_mean", ascending=False
        )


class TimeSeriesSplit:
    """
    Custom time series split for walk-forward validation.

    Similar to sklearn's TimeSeriesSplit but with configurable
    test size and gap between train/test.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        """
        Initialize time series split.

        Args:
            n_splits: Number of splits.
            test_size: Size of test set (if None, calculated automatically).
            gap: Gap between train and test sets.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for train/test splits.

        Args:
            X: Features DataFrame.
            y: Target Series (unused, for API compatibility).

        Yields:
            Tuples of (train_indices, test_indices).
        """
        n_samples = len(X)

        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        splits = []

        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            train_end = test_start - self.gap

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            splits.append((train_indices, test_indices))

        return splits

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits
