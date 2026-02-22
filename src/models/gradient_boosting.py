"""Gradient Boosting model implementations."""

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseModel, ModelFactory
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM model for regression."""

    def __init__(
        self,
        name: str = "LightGBM",
        params: dict[str, Any] | None = None,
    ):
        """
        Initialize LightGBM model.

        Args:
            name: Model name.
            params: Model parameters.
        """
        if params is None:
            params = Config().get_model_params("lightgbm")
        super().__init__(name, params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LightGBMModel":
        """Train the LightGBM model."""
        self.feature_names = X_train.columns.tolist()

        self.model = lgb.LGBMRegressor(**self.params)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )

            y_pred_val = self.model.predict(X_val)
            self.training_metrics = self._calculate_metrics(y_val.values, y_pred_val)
        else:
            self.model.fit(X_train, y_train)

            y_pred_train = self.model.predict(X_train)
            self.training_metrics = self._calculate_metrics(y_train.values, y_pred_train)

        self.is_fitted = True
        logger.info(f"LightGBM trained: RMSE={self.training_metrics.get('rmse', 'N/A'):.6f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)


class XGBoostModel(BaseModel):
    """XGBoost model for regression."""

    def __init__(
        self,
        name: str = "XGBoost",
        params: dict[str, Any] | None = None,
    ):
        """
        Initialize XGBoost model.

        Args:
            name: Model name.
            params: Model parameters.
        """
        if params is None:
            params = Config().get_model_params("xgboost")
        super().__init__(name, params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "XGBoostModel":
        """Train the XGBoost model."""
        self.feature_names = X_train.columns.tolist()

        self.model = xgb.XGBRegressor(**self.params)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred_val = self.model.predict(X_val)
            self.training_metrics = self._calculate_metrics(y_val.values, y_pred_val)
        else:
            self.model.fit(X_train, y_train)

            y_pred_train = self.model.predict(X_train)
            self.training_metrics = self._calculate_metrics(y_train.values, y_pred_train)

        self.is_fitted = True
        logger.info(f"XGBoost trained: RMSE={self.training_metrics.get('rmse', 'N/A'):.6f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)


class CatBoostModel(BaseModel):
    """CatBoost model for regression."""

    def __init__(
        self,
        name: str = "CatBoost",
        params: dict[str, Any] | None = None,
    ):
        """
        Initialize CatBoost model.

        Args:
            name: Model name.
            params: Model parameters.
        """
        if params is None:
            params = Config().get_model_params("catboost")
        super().__init__(name, params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "CatBoostModel":
        """Train the CatBoost model."""
        try:
            from catboost import CatBoostRegressor
        except ImportError as err:
            raise ImportError("CatBoost not installed. Install with: pip install catboost") from err

        self.feature_names = X_train.columns.tolist()

        # Remove early_stopping_rounds from params if present (pass separately)
        params = self.params.copy()
        early_stopping = params.pop("early_stopping_rounds", 50)

        self.model = CatBoostRegressor(**params)

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=early_stopping,
            )

            y_pred_val = self.model.predict(X_val)
            self.training_metrics = self._calculate_metrics(y_val.values, y_pred_val)
        else:
            self.model.fit(X_train, y_train)

            y_pred_train = self.model.predict(X_train)
            self.training_metrics = self._calculate_metrics(y_train.values, y_pred_train)

        self.is_fitted = True
        logger.info(f"CatBoost trained: RMSE={self.training_metrics.get('rmse', 'N/A'):.6f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)


class RandomForestModel(BaseModel):
    """Random Forest model for regression."""

    def __init__(
        self,
        name: str = "RandomForest",
        params: dict[str, Any] | None = None,
    ):
        """
        Initialize Random Forest model.

        Args:
            name: Model name.
            params: Model parameters.
        """
        if params is None:
            params = Config().get_model_params("random_forest")
        super().__init__(name, params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "RandomForestModel":
        """Train the Random Forest model."""
        self.feature_names = X_train.columns.tolist()

        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            self.training_metrics = self._calculate_metrics(y_val.values, y_pred_val)
        else:
            y_pred_train = self.model.predict(X_train)
            self.training_metrics = self._calculate_metrics(y_train.values, y_pred_train)

        self.is_fitted = True
        logger.info(
            f"RandomForest trained: RMSE={self.training_metrics.get('rmse', 'N/A'):.6f}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)


# Register models with factory
ModelFactory.register("lightgbm", LightGBMModel)
ModelFactory.register("xgboost", XGBoostModel)
ModelFactory.register("catboost", CatBoostModel)
ModelFactory.register("random_forest", RandomForestModel)
