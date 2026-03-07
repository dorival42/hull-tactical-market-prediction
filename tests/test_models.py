"""Tests for model implementations."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.base import ModelFactory
from src.models.ensemble import EnsembleModel
from src.models.gradient_boosting import (
    LightGBMModel,
    RandomForestModel,
    XGBoostModel,
)


class TestLightGBMModel:
    """Tests for LightGBM model."""

    def test_init(self, model_params_lightgbm):
        """Test initialization."""
        model = LightGBMModel(params=model_params_lightgbm)
        assert model.name == "LightGBM"
        assert not model.is_fitted
        assert model.params == model_params_lightgbm

    def test_fit(self, sample_features, sample_target, model_params_lightgbm):
        """Test model fitting."""
        model = LightGBMModel(params=model_params_lightgbm)
        model.fit(sample_features, sample_target)

        assert model.is_fitted
        assert model.feature_names is not None
        assert len(model.feature_names) == sample_features.shape[1]

    def test_fit_with_validation(self, sample_features, sample_target, model_params_lightgbm):
        """Test model fitting with validation set."""
        # Split data
        split = int(len(sample_features) * 0.8)
        X_train = sample_features.iloc[:split]
        y_train = sample_target.iloc[:split]
        X_val = sample_features.iloc[split:]
        y_val = sample_target.iloc[split:]

        model = LightGBMModel(params=model_params_lightgbm)
        model.fit(X_train, y_train, X_val, y_val)

        assert model.is_fitted
        assert "rmse" in model.training_metrics

    def test_predict(self, sample_features, sample_target, model_params_lightgbm):
        """Test prediction."""
        model = LightGBMModel(params=model_params_lightgbm)
        model.fit(sample_features, sample_target)

        predictions = model.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert not np.isnan(predictions).any()

    def test_get_feature_importance(self, sample_features, sample_target, model_params_lightgbm):
        """Test feature importance."""
        model = LightGBMModel(params=model_params_lightgbm)
        model.fit(sample_features, sample_target)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == sample_features.shape[1]

    def test_save_load(self, sample_features, sample_target, model_params_lightgbm):
        """Test model save and load."""
        model = LightGBMModel(params=model_params_lightgbm)
        model.fit(sample_features, sample_target)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            model.save(filepath)

            loaded_model = LightGBMModel()
            loaded_model.load(filepath)

            assert loaded_model.is_fitted
            assert loaded_model.name == model.name

            # Check predictions are same
            orig_pred = model.predict(sample_features)
            load_pred = loaded_model.predict(sample_features)
            np.testing.assert_array_almost_equal(orig_pred, load_pred)

        finally:
            Path(filepath).unlink(missing_ok=True)


class TestXGBoostModel:
    """Tests for XGBoost model."""

    def test_init(self, model_params_xgboost):
        """Test initialization."""
        model = XGBoostModel(params=model_params_xgboost)
        assert model.name == "XGBoost"
        assert not model.is_fitted

    def test_fit_predict(self, sample_features, sample_target, model_params_xgboost):
        """Test fitting and prediction."""
        model = XGBoostModel(params=model_params_xgboost)
        model.fit(sample_features, sample_target)

        predictions = model.predict(sample_features)

        assert model.is_fitted
        assert len(predictions) == len(sample_features)


class TestRandomForestModel:
    """Tests for Random Forest model."""

    def test_init(self):
        """Test initialization."""
        model = RandomForestModel()
        assert model.name == "RandomForest"
        assert not model.is_fitted

    def test_fit_predict(self, sample_features, sample_target):
        """Test fitting and prediction."""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
            "n_jobs": -1,
        }
        model = RandomForestModel(params=params)
        model.fit(sample_features, sample_target)

        predictions = model.predict(sample_features)

        assert model.is_fitted
        assert len(predictions) == len(sample_features)


class TestEnsembleModel:
    """Tests for Ensemble model."""

    def test_init(self):
        """Test initialization."""
        ensemble = EnsembleModel()
        assert ensemble.name == "Ensemble"
        assert len(ensemble.models) == 0

    def test_add_model(self, model_params_lightgbm):
        """Test adding models to ensemble."""
        ensemble = EnsembleModel()
        model = LightGBMModel(params=model_params_lightgbm)

        ensemble.add_model(model, weight=1.0)

        assert len(ensemble.models) == 1
        assert ensemble.weights["LightGBM"] == 1.0

    def test_fit_predict(self, sample_features, sample_target, model_params_lightgbm):
        """Test ensemble fitting and prediction."""
        # Create simple ensemble with one model for testing
        ensemble = EnsembleModel()
        model = LightGBMModel(params=model_params_lightgbm)
        ensemble.add_model(model, weight=1.0)

        ensemble.fit(sample_features, sample_target)

        predictions = ensemble.predict(sample_features)

        assert ensemble.is_fitted
        assert len(predictions) == len(sample_features)

    def test_get_feature_importance(self, sample_features, sample_target, model_params_lightgbm):
        """Test aggregated feature importance."""
        ensemble = EnsembleModel()
        model = LightGBMModel(params=model_params_lightgbm)
        ensemble.add_model(model, weight=1.0)

        ensemble.fit(sample_features, sample_target)

        importance = ensemble.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns


class TestModelFactory:
    """Tests for ModelFactory."""

    def test_create_lightgbm(self):
        """Test creating LightGBM model."""
        model = ModelFactory.create("lightgbm")
        assert isinstance(model, LightGBMModel)

    def test_create_xgboost(self):
        """Test creating XGBoost model."""
        model = ModelFactory.create("xgboost")
        assert isinstance(model, XGBoostModel)

    def test_create_random_forest(self):
        """Test creating Random Forest model."""
        model = ModelFactory.create("random_forest")
        assert isinstance(model, RandomForestModel)

    def test_create_unknown_raises(self):
        """Test that unknown model type raises error."""
        with pytest.raises(ValueError):
            ModelFactory.create("unknown_model")

    def test_available_models(self):
        """Test getting available model types."""
        available = ModelFactory.available_models()
        assert "lightgbm" in available
        assert "xgboost" in available
