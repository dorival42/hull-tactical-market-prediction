"""Inference predictor for Hull Tactical."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_engineer import FeatureEngineer
from src.models.base import BaseModel
from src.models.ensemble import EnsembleModel
from src.models.registry import ModelRegistry
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Inference predictor for making predictions.

    Handles:
    - Model loading (local or from registry)
    - Feature engineering on new data
    - Prediction generation
    """

    def __init__(
        self,
        model: BaseModel | None = None,
        model_path: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ):
        """
        Initialize predictor.

        Args:
            model: Pre-loaded model instance.
            model_path: Path to local model file.
            model_name: Name of model in registry.
            model_version: Version of model in registry.
        """
        self.config = Config()
        self.model: BaseModel | None = model
        self.feature_engineer = FeatureEngineer(verbose=False)
        self.feature_cols: list[str] = []

        # Load model if path or name provided
        if model_path:
            self.load_model_local(model_path)
        elif model_name:
            self.load_model_registry(model_name, model_version)

    def load_model_local(self, model_path: str) -> None:
        """
        Load model from local file.

        Args:
            model_path: Path to model file.
        """
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Determine model type from filename
        filename = path.stem.lower()

        if "ensemble" in filename:
            self.model = EnsembleModel()
        elif "lightgbm" in filename:
            from src.models.gradient_boosting import LightGBMModel

            self.model = LightGBMModel()
        elif "xgboost" in filename:
            from src.models.gradient_boosting import XGBoostModel

            self.model = XGBoostModel()
        elif "catboost" in filename:
            from src.models.gradient_boosting import CatBoostModel

            self.model = CatBoostModel()
        elif "random_forest" in filename or "rf" in filename:
            from src.models.gradient_boosting import RandomForestModel

            self.model = RandomForestModel()
        else:
            # Try to load as generic model
            from src.models.gradient_boosting import LightGBMModel

            self.model = LightGBMModel()

        self.model.load(model_path)
        self.feature_cols = self.model.feature_names or []

        logger.info(f"Loaded model from {model_path}")

    def load_model_registry(
        self,
        model_name: str,
        version: str | None = None,
        stage: str | None = None,
    ) -> None:
        """
        Load model from MLflow registry.

        Args:
            model_name: Registered model name.
            version: Specific version to load.
            stage: Stage to load from ('Production', 'Staging').
        """
        registry = ModelRegistry()
        self.model = registry.load_model(model_name, version=version, stage=stage)
        logger.info(f"Loaded model {model_name} from registry")

    def load_features(self, features_path: str) -> None:
        """
        Load feature list from JSON file.

        Args:
            features_path: Path to features JSON file.
        """
        with open(features_path) as f:
            self.feature_cols = json.load(f)
        logger.info(f"Loaded {len(self.feature_cols)} features from {features_path}")

    def predict(
        self,
        data: pd.DataFrame | dict[str, Any],
    ) -> dict[str, Any]:
        """
        Make predictions on new data.

        Args:
            data: Input data (DataFrame or dict).

        Returns:
            Dictionary with predictions.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model_* first.")

        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Apply feature engineering
        data_transformed = self.feature_engineer.transform(data)

        # Select features
        if self.feature_cols:
            missing_cols = set(self.feature_cols) - set(data_transformed.columns)
            for col in missing_cols:
                data_transformed[col] = 0

            X = data_transformed[self.feature_cols].fillna(0)
        else:
            # Use all numeric columns
            X = data_transformed.select_dtypes(include=["float64", "int64"]).fillna(0)

        # Make predictions
        predictions = self.model.predict(X)

        return {
            "predictions": predictions.tolist(),
            "n_samples": len(predictions),
        }

    def predict_single(
        self,
        data: pd.Series | dict[str, Any],
    ) -> dict[str, float]:
        """
        Make prediction for a single sample.

        Args:
            data: Single sample data.

        Returns:
            Dictionary with prediction.
        """
        if isinstance(data, pd.Series):
            data = data.to_dict()

        result = self.predict(data)

        return {
            "prediction": result["predictions"][0],
        }

    def batch_predict(
        self,
        data: pd.DataFrame,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """
        Make predictions in batches.

        Args:
            data: Input DataFrame.
            batch_size: Size of each batch.

        Returns:
            DataFrame with predictions.
        """
        all_predictions = []

        for i in range(0, len(data), batch_size):
            batch = data.iloc[i : i + batch_size]
            result = self.predict(batch)
            all_predictions.extend(result["predictions"])

        return pd.DataFrame({"prediction": all_predictions})

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information.
        """
        if self.model is None:
            return {"status": "No model loaded"}

        return {
            "name": self.model.name,
            "is_fitted": self.model.is_fitted,
            "n_features": len(self.feature_cols),
            "params": self.model.get_params(),
            "training_metrics": self.model.get_metrics(),
        }


class InferenceServer:
    """
    gRPC inference server for Kaggle competition.

    This class wraps the Predictor for use with the Kaggle
    evaluation framework.
    """

    def __init__(
        self,
        model_path: str,
        features_path: str | None = None,
    ):
        """
        Initialize inference server.

        Args:
            model_path: Path to model file.
            features_path: Path to features JSON file.
        """
        self.predictor = Predictor(model_path=model_path)

        if features_path:
            self.predictor.load_features(features_path)

    def predict(self, features: dict[str, Any]) -> float:
        """
        Make a single prediction.

        Args:
            features: Feature dictionary.

        Returns:
            Prediction value.
        """
        result = self.predictor.predict_single(features)
        return result["prediction"]

    def batch_predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make batch predictions.

        Args:
            data: Input DataFrame.

        Returns:
            Array of predictions.
        """
        result = self.predictor.batch_predict(data)
        return result["prediction"].values
