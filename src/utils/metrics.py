"""Metrics calculation for Hull Tactical."""


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelMetrics:
    """Class for calculating model performance metrics."""

    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            Dictionary with RMSE, MAE, R2, and MAPE.
        """
        # Filter NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = np.asarray(y_true)[mask]
        y_pred_clean = np.asarray(y_pred)[mask]

        if len(y_true_clean) == 0:
            return {
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "mape": np.nan,
            }

        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)

        # MAPE (avoid division by zero)
        mask_nonzero = y_true_clean != 0
        if mask_nonzero.sum() > 0:
            mape = (
                np.mean(
                    np.abs(
                        (y_true_clean[mask_nonzero] - y_pred_clean[mask_nonzero])
                        / y_true_clean[mask_nonzero]
                    )
                )
                * 100
            )
        else:
            mape = np.nan

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
        }

    @staticmethod
    def calculate_directional_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Calculate directional accuracy (percentage of correct sign predictions).

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            Directional accuracy as percentage.
        """
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = np.asarray(y_true)[mask]
        y_pred_clean = np.asarray(y_pred)[mask]

        if len(y_true_clean) == 0:
            return np.nan

        correct = np.sum(np.sign(y_true_clean) == np.sign(y_pred_clean))
        return float(correct / len(y_true_clean) * 100)

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate all prediction metrics.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            Dictionary with all metrics.
        """
        metrics = ModelMetrics.calculate_regression_metrics(y_true, y_pred)
        metrics["directional_accuracy"] = ModelMetrics.calculate_directional_accuracy(
            y_true, y_pred
        )
        return metrics
