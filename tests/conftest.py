"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_train_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 1000

    data = {
        "date_id": range(n_samples),
        "forward_returns": np.random.randn(n_samples) * 0.01,
        "risk_free_rate": np.random.uniform(0.01, 0.05, n_samples),
    }

    # Add feature columns
    for prefix in ["D", "E", "I", "M", "P", "S", "V"]:
        for i in range(5):
            col_name = f"{prefix}{i+1}"
            data[col_name] = np.random.randn(n_samples)

    df = pd.DataFrame(data)

    # Calculate target
    df["market_forward_excess_returns"] = df["forward_returns"] - df["risk_free_rate"] / 252

    return df


@pytest.fixture
def sample_test_data():
    """Generate sample test data."""
    np.random.seed(123)
    n_samples = 10

    data = {
        "date_id": range(8990, 8990 + n_samples),
        "lagged_forward_returns": np.random.randn(n_samples) * 0.01,
        "lagged_risk_free_rate": np.random.uniform(0.01, 0.05, n_samples),
        "lagged_market_forward_excess_returns": np.random.randn(n_samples) * 0.01,
        "is_scored": [True] * n_samples,
    }

    # Add feature columns
    for prefix in ["D", "E", "I", "M", "P", "S", "V"]:
        for i in range(5):
            col_name = f"{prefix}{i+1}"
            data[col_name] = np.random.randn(n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Generate sample feature matrix."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )

    return X


@pytest.fixture
def sample_target():
    """Generate sample target vector."""
    np.random.seed(42)
    n_samples = 100

    return pd.Series(np.random.randn(n_samples) * 0.01, name="target")


@pytest.fixture
def sample_predictions():
    """Generate sample predictions."""
    np.random.seed(42)
    n_samples = 100

    return np.random.randn(n_samples) * 0.01


@pytest.fixture
def model_params_lightgbm():
    """LightGBM model parameters for testing."""
    return {
        "n_estimators": 50,
        "learning_rate": 0.1,
        "max_depth": 3,
        "num_leaves": 15,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
    }


@pytest.fixture
def model_params_xgboost():
    """XGBoost model parameters for testing."""
    return {
        "n_estimators": 50,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    }


@pytest.fixture
def sample_features_with_nan():
    """Generate sample feature matrix with missing values."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )

    # Add 10% missing values randomly
    mask = np.random.random(X.shape) < 0.1
    X = X.mask(mask)

    return X


@pytest.fixture
def sample_data_with_high_nan():
    """Generate sample data with some columns having high NaN percentage."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "good_col_1": np.random.randn(n_samples),
        "good_col_2": np.random.randn(n_samples),
        "good_col_3": np.random.randn(n_samples),
        "high_nan_col": np.random.randn(n_samples),  # Will have 50% NaN
        "very_high_nan_col": np.random.randn(n_samples),  # Will have 80% NaN
    }

    df = pd.DataFrame(data)

    # Add missing values
    df.loc[:49, "high_nan_col"] = np.nan  # 50% NaN
    df.loc[:79, "very_high_nan_col"] = np.nan  # 80% NaN

    return df
