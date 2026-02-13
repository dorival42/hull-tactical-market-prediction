"""Ensemble model implementations."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseModel, ModelFactory
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple base models.

    Supports:
    - Weighted averaging
    - Simple averaging
    - Stacking (optional)
    """

    def __init__(
        self,
        name: str = "Ensemble",
        models: Optional[List[BaseModel]] = None,
        weights: Optional[Dict[str, float]] = None,
        method: str = "weighted_average",
    ):
        """
        Initialize ensemble model.

        Args:
            name: Model name.
            models: List of base models.
            weights: Dictionary mapping model names to weights.
            method: Ensemble method ('weighted_average', 'simple_average').
        """
        super().__init__(name, params={})
        self.models = models or []
        self.weights = weights or {}
        self.method = method

        if not self.weights and self.models:
            # Default: equal weights
            n_models = len(self.models)
            self.weights = {m.name: 1.0 / n_models for m in self.models}

    @classmethod
    def from_config(
        cls,
        model_types: Optional[List[str]] = None,
        name: str = "Ensemble",
    ) -> "EnsembleModel":
        """
        Create ensemble from configuration.

        Args:
            model_types: List of model types to include.
            name: Ensemble name.

        Returns:
            EnsembleModel instance.
        """
        config = Config()
        ensemble_config = config.model_params.get("ensemble", {})

        if model_types is None:
            model_types = ["lightgbm", "xgboost", "catboost", "random_forest"]

        models = []
        for model_type in model_types:
            try:
                model = ModelFactory.create(model_type)
                models.append(model)
            except Exception as e:
                logger.warning(f"Could not create {model_type}: {e}")

        weights = ensemble_config.get("weights", {})
        method = ensemble_config.get("method", "weighted_average")

        return cls(name=name, models=models, weights=weights, method=method)

    def add_model(self, model: BaseModel, weight: float = 1.0) -> "EnsembleModel":
        """
        Add a model to the ensemble.

        Args:
            model: Model to add.
            weight: Weight for this model.

        Returns:
            Self for chaining.
        """
        self.models.append(model)
        self.weights[model.name] = weight
        return self

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "EnsembleModel":
        """
        Train all models in the ensemble.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.

        Returns:
            Self for chaining.
        """
        self.feature_names = X_train.columns.tolist()

        for model in self.models:
            logger.info(f"Training {model.name}...")
            model.fit(X_train, y_train, X_val, y_val)

        # Calculate ensemble metrics on validation set
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            self.training_metrics = self._calculate_metrics(y_val.values, y_pred)
            logger.info(
                f"Ensemble RMSE: {self.training_metrics.get('rmse', 'N/A'):.6f}"
            )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Features for prediction.

        Returns:
            Array of predictions.
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        weights = []

        for model in self.models:
            if model.is_fitted:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.weights.get(model.name, 1.0))

        if not predictions:
            raise ValueError("No fitted models in ensemble")

        predictions = np.array(predictions)
        weights = np.array(weights)

        if self.method == "weighted_average":
            # Normalize weights
            weights = weights / weights.sum()
            return np.average(predictions, axis=0, weights=weights)

        elif self.method == "simple_average":
            return np.mean(predictions, axis=0)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get aggregated feature importance from all models.

        Returns:
            DataFrame with feature importance.
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        all_importance = []

        for model in self.models:
            if model.is_fitted:
                imp = model.get_feature_importance()
                imp["model"] = model.name
                imp["weight"] = self.weights.get(model.name, 1.0)
                all_importance.append(imp)

        if not all_importance:
            raise ValueError("No fitted models in ensemble")

        combined = pd.concat(all_importance, ignore_index=True)

        # Aggregate by weighted average
        aggregated = (
            combined.groupby("feature")
            .apply(
                lambda x: np.average(x["importance"], weights=x["weight"])
            )
            .reset_index()
        )
        aggregated.columns = ["feature", "importance"]
        aggregated = aggregated.sort_values("importance", ascending=False)

        return aggregated

    def get_model_weights(self) -> Dict[str, float]:
        """Get normalized model weights."""
        total = sum(self.weights.values())
        return {k: v / total for k, v in self.weights.items()}

    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.

        Args:
            X: Features for prediction.

        Returns:
            Dictionary mapping model names to predictions.
        """
        predictions = {}
        for model in self.models:
            if model.is_fitted:
                predictions[model.name] = model.predict(X)
        return predictions

    def save(self, filepath: str) -> None:
        """Save ensemble and all models."""
        import pickle
        from pathlib import Path

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        ensemble_data = {
            "name": self.name,
            "weights": self.weights,
            "method": self.method,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "models": [],
        }

        for model in self.models:
            ensemble_data["models"].append(
                {
                    "name": model.name,
                    "class": model.__class__.__name__,
                    "params": model.params,
                    "model": model.model,
                    "feature_names": model.feature_names,
                    "training_metrics": model.training_metrics,
                }
            )

        with open(path, "wb") as f:
            pickle.dump(ensemble_data, f)

        logger.info(f"Ensemble saved: {filepath}")

    def load(self, filepath: str) -> "EnsembleModel":
        """Load ensemble and all models."""
        import pickle

        with open(filepath, "rb") as f:
            ensemble_data = pickle.load(f)

        self.name = ensemble_data["name"]
        self.weights = ensemble_data["weights"]
        self.method = ensemble_data["method"]
        self.feature_names = ensemble_data["feature_names"]
        self.training_metrics = ensemble_data["training_metrics"]

        # Reconstruct models
        model_classes = {
            "LightGBMModel": "src.models.gradient_boosting.LightGBMModel",
            "XGBoostModel": "src.models.gradient_boosting.XGBoostModel",
            "CatBoostModel": "src.models.gradient_boosting.CatBoostModel",
            "RandomForestModel": "src.models.gradient_boosting.RandomForestModel",
        }

        self.models = []
        for model_data in ensemble_data["models"]:
            # Create model instance
            model_type = model_data["class"].lower().replace("model", "")
            if model_type == "randomforest":
                model_type = "random_forest"

            model = ModelFactory.create(
                model_type,
                name=model_data["name"],
                params=model_data["params"],
            )
            model.model = model_data["model"]
            model.feature_names = model_data["feature_names"]
            model.training_metrics = model_data["training_metrics"]
            model.is_fitted = True

            self.models.append(model)

        self.is_fitted = True
        logger.info(f"Ensemble loaded: {filepath}")
        return self


# Register ensemble with factory
ModelFactory.register("ensemble", EnsembleModel)
