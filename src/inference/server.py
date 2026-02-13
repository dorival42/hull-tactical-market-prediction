"""gRPC server for Kaggle inference."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.features.feature_engineer import FeatureEngineer
from src.inference.predictor import Predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KaggleInferenceServer:
    """
    Inference server compatible with Kaggle evaluation framework.

    This server is designed to work with the Hull Tactical
    competition's gRPC-based evaluation system.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        features_path: Optional[str] = None,
        artifacts_dir: str = "artifacts",
    ):
        """
        Initialize Kaggle inference server.

        Args:
            model_path: Path to model file.
            features_path: Path to features JSON file.
            artifacts_dir: Directory containing model artifacts.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.predictor: Optional[Predictor] = None
        self.feature_engineer = FeatureEngineer(verbose=False)
        self.feature_cols: list = []

        # Auto-load if paths provided
        if model_path:
            self._load_model(model_path)

        if features_path:
            self._load_features(features_path)
        elif self.artifacts_dir.exists():
            # Try to auto-load features
            default_features = self.artifacts_dir / "selected_features.json"
            if default_features.exists():
                self._load_features(str(default_features))

    def _load_model(self, model_path: str) -> None:
        """Load model from path."""
        self.predictor = Predictor(model_path=model_path)
        logger.info(f"Loaded model: {model_path}")

    def _load_features(self, features_path: str) -> None:
        """Load feature list from JSON."""
        with open(features_path, "r") as f:
            self.feature_cols = json.load(f)
        logger.info(f"Loaded {len(self.feature_cols)} features")

    def initialize(self) -> None:
        """
        Initialize the server with default artifacts.

        Called by the Kaggle evaluation framework.
        """
        if self.predictor is not None:
            return

        # Find best model in artifacts
        model_files = list(self.artifacts_dir.glob("*_final.pkl"))

        if not model_files:
            raise FileNotFoundError(
                f"No model files found in {self.artifacts_dir}"
            )

        # Prefer ensemble, then lightgbm, then any
        for pattern in ["ensemble", "lightgbm", "xgboost"]:
            for mf in model_files:
                if pattern in mf.stem.lower():
                    self._load_model(str(mf))
                    return

        # Load first available
        self._load_model(str(model_files[0]))

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Make prediction for a single observation.

        This method is called by the Kaggle gRPC framework.

        Args:
            features: Dictionary of feature values.

        Returns:
            Predicted market forward excess return.
        """
        if self.predictor is None:
            self.initialize()

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Apply feature engineering
        df_transformed = self.feature_engineer.transform(df)

        # Select features
        if self.feature_cols:
            # Add missing columns with zeros
            for col in self.feature_cols:
                if col not in df_transformed.columns:
                    df_transformed[col] = 0

            X = df_transformed[self.feature_cols].fillna(0)
        else:
            # Use model's feature names
            X = df_transformed[self.predictor.model.feature_names].fillna(0)

        # Get prediction
        prediction = self.predictor.model.predict(X)[0]

        return float(prediction)

    def batch_predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for a batch of observations.

        Args:
            df: DataFrame with features.

        Returns:
            Array of predictions.
        """
        if self.predictor is None:
            self.initialize()

        # Apply feature engineering
        df_transformed = self.feature_engineer.transform(df)

        # Select features
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in df_transformed.columns:
                    df_transformed[col] = 0

            X = df_transformed[self.feature_cols].fillna(0)
        else:
            X = df_transformed[self.predictor.model.feature_names].fillna(0)

        # Get predictions
        predictions = self.predictor.model.predict(X)

        return predictions


# Factory function for Kaggle framework
def create_inference_server(
    model_path: Optional[str] = None,
    **kwargs,
) -> KaggleInferenceServer:
    """
    Create an inference server instance.

    Args:
        model_path: Path to model file.
        **kwargs: Additional arguments.

    Returns:
        Configured inference server.
    """
    return KaggleInferenceServer(model_path=model_path, **kwargs)
