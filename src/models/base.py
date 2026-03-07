"""Base model class for Hull Tactical."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.metrics import ModelMetrics

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all models.

    All models must implement:
    - fit(): Train the model
    - predict(): Make predictions
    - get_feature_importance(): Get feature importance scores
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        """
        Initialize model.

        Args:
            name: Model name.
            params: Model parameters dictionary.
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names: list | None = None
        self.training_metrics: dict[str, float] = {}

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "BaseModel":
        """
        Train the model.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features (optional).
            y_val: Validation target (optional).

        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features for prediction.

        Returns:
            Array of predictions.
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with columns ['feature', 'importance'].
        """
        pass

    def save(self, filepath: str) -> None:
        """
        Save the model to a file.

        Args:
            filepath: Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "name": self.name,
            "params": self.params,
            "model": self.model,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved: {filepath}")

    def load(self, filepath: str) -> "BaseModel":
        """
        Load the model from a file.

        Args:
            filepath: Path to the model file.

        Returns:
            Self for chaining.
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.name = model_data["name"]
        self.params = model_data["params"]
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.training_metrics = model_data["training_metrics"]
        self.is_fitted = True

        logger.info(f"Model loaded: {filepath}")
        return self

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()

    def set_params(self, **params) -> "BaseModel":
        """Set model parameters."""
        self.params.update(params)
        return self

    def get_metrics(self) -> dict[str, float]:
        """Get training metrics."""
        return self.training_metrics.copy()

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate regression metrics."""
        return ModelMetrics.calculate_regression_metrics(y_true, y_pred)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"


class ModelFactory:
    """Factory class for creating models."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, model_type: str, model_class: type) -> None:
        """Register a model class."""
        cls._registry[model_type] = model_class

    @classmethod
    def create(
        cls,
        model_type: str,
        name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> BaseModel:
        """
        Create a model instance.

        Args:
            model_type: Type of model ('lightgbm', 'xgboost', etc.).
            name: Optional model name.
            params: Optional model parameters.

        Returns:
            Model instance.
        """
        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown model type: {model_type}. " f"Available: {list(cls._registry.keys())}"
            )

        if params is None:
            config = Config()
            params = config.get_model_params(model_type)

        model_class = cls._registry[model_type]
        return model_class(name=name or model_type, params=params)

    @classmethod
    def available_models(cls) -> list:
        """Get list of available model types."""
        return list(cls._registry.keys())
