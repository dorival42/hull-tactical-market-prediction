"""Tests for data loading and preprocessing."""


import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    AdvancedFeatureSelector,
    CatBoostImputer,
    DataPipeline,
    DataPreprocessor,
)


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_init(self):
        """Test initialization."""
        preprocessor = DataPreprocessor(verbose=False)
        assert preprocessor.nan_threshold == 0.30
        assert preprocessor.use_catboost_imputation

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        preprocessor = DataPreprocessor(
            nan_threshold=0.25,
            use_catboost_imputation=False,
            catboost_iterations=50,
            verbose=False,
        )
        assert preprocessor.nan_threshold == 0.25
        assert not preprocessor.use_catboost_imputation
        assert preprocessor.catboost_iterations == 50

    def test_fit(self, sample_train_data):
        """Test fit method."""
        preprocessor = DataPreprocessor(verbose=False)
        preprocessor.fit(sample_train_data)

        assert preprocessor._fitted
        assert len(preprocessor._kept_columns) > 0

    def test_transform(self, sample_train_data):
        """Test transform method."""
        preprocessor = DataPreprocessor(use_catboost_imputation=False, verbose=False)
        preprocessor.fit(sample_train_data)

        # Add some NaN values
        df = sample_train_data.copy()
        df.loc[0:10, "D1"] = np.nan

        transformed = preprocessor.transform(df)

        # Should not have NaN after transform
        numeric_cols = transformed.select_dtypes(include=["float64", "int64"]).columns
        assert not transformed[numeric_cols].isna().any().any()

    def test_fit_transform(self, sample_train_data):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor(use_catboost_imputation=False, verbose=False)
        transformed = preprocessor.fit_transform(sample_train_data)

        assert transformed.shape[0] == sample_train_data.shape[0]

    def test_drop_high_nan_columns(self, sample_train_data):
        """Test that high NaN columns are dropped."""
        preprocessor = DataPreprocessor(nan_threshold=0.30, verbose=False)

        # Create column with >30% NaN
        df = sample_train_data.copy()
        df["high_nan_col"] = np.nan
        df.loc[: len(df) // 2, "high_nan_col"] = 1.0  # 50% NaN

        preprocessor.fit(df)
        transformed = preprocessor.transform(df)

        assert "high_nan_col" in preprocessor._dropped_columns
        assert "high_nan_col" not in transformed.columns

    def test_get_dropped_columns(self, sample_train_data):
        """Test getting dropped columns list."""
        preprocessor = DataPreprocessor(nan_threshold=0.30, verbose=False)

        # Create column with >30% NaN
        df = sample_train_data.copy()
        df["high_nan_col"] = np.nan
        df.loc[: len(df) // 2, "high_nan_col"] = 1.0

        preprocessor.fit(df)
        dropped = preprocessor.get_dropped_columns()

        assert "high_nan_col" in dropped

    def test_get_kept_columns(self, sample_train_data):
        """Test getting kept columns list."""
        preprocessor = DataPreprocessor(nan_threshold=0.30, verbose=False)
        preprocessor.fit(sample_train_data)

        kept = preprocessor.get_kept_columns()

        assert len(kept) > 0
        assert "D1" in kept

    def test_handle_inf_values(self, sample_train_data):
        """Test that inf values are handled."""
        preprocessor = DataPreprocessor(use_catboost_imputation=False, verbose=False)

        # Add inf values
        df = sample_train_data.copy()
        df.loc[0, "D1"] = np.inf
        df.loc[1, "E1"] = -np.inf

        preprocessor.fit(df)
        transformed = preprocessor.transform(df)

        numeric_cols = transformed.select_dtypes(include=["float64", "int64"]).columns
        assert not np.isinf(transformed[numeric_cols]).any().any()

    def test_get_preprocessing_report(self, sample_train_data):
        """Test preprocessing report generation."""
        preprocessor = DataPreprocessor(verbose=False)

        # Add some NaN values
        df = sample_train_data.copy()
        df.loc[0:50, "D1"] = np.nan

        report = preprocessor.get_preprocessing_report(df)

        assert "column" in report.columns
        assert "missing_pct" in report.columns
        assert "will_drop" in report.columns


class TestMetrics:
    """Tests for metrics calculation."""

    def test_regression_metrics(self):
        """Test regression metrics calculation."""
        from src.utils.metrics import ModelMetrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

        metrics = ModelMetrics.calculate_regression_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

    def test_directional_accuracy(self):
        """Test directional accuracy calculation."""
        from src.utils.metrics import ModelMetrics

        y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        y_pred = np.array([0.02, -0.01, 0.01, 0.01, 0.03])

        accuracy = ModelMetrics.calculate_directional_accuracy(y_true, y_pred)

        assert 0 <= accuracy <= 100

    def test_calculate_all_metrics(self):
        """Test calculate_all_metrics combines regression and directional."""
        from src.utils.metrics import ModelMetrics

        y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        y_pred = np.array([0.02, -0.01, 0.01, 0.01, 0.03])

        metrics = ModelMetrics.calculate_all_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "directional_accuracy" in metrics


class TestCatBoostImputer:
    """Tests for CatBoost-based imputation."""

    def test_init(self):
        """Test initialization."""
        imputer = CatBoostImputer()
        assert imputer.iterations == 100
        assert imputer.learning_rate == 0.1
        assert imputer.depth == 4

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        imputer = CatBoostImputer(iterations=50, learning_rate=0.05, depth=3)
        assert imputer.iterations == 50
        assert imputer.learning_rate == 0.05
        assert imputer.depth == 3

    def test_fit_transform(self, sample_features):
        """Test fit_transform on data with missing values."""
        imputer = CatBoostImputer(iterations=10, verbose=False)

        # Add missing values
        df = sample_features.copy()
        np.random.seed(42)
        mask = np.random.random(df.shape) < 0.1  # 10% missing
        df = df.mask(mask)

        # Fit and transform
        result = imputer.fit_transform(df)

        # Check no NaN values remain
        assert not result.isna().any().any()
        assert result.shape == df.shape

    def test_fit_then_transform(self, sample_features):
        """Test separate fit and transform."""
        imputer = CatBoostImputer(iterations=10, verbose=False)

        # Add missing values
        df = sample_features.copy()
        np.random.seed(42)
        mask = np.random.random(df.shape) < 0.1
        df = df.mask(mask)

        # Fit
        imputer.fit(df)
        assert len(imputer.models) > 0

        # Transform
        result = imputer.transform(df)
        assert not result.isna().any().any()

    def test_no_missing_values(self, sample_features):
        """Test with data that has no missing values."""
        imputer = CatBoostImputer(iterations=10, verbose=False)

        result = imputer.fit_transform(sample_features)

        # Should return data unchanged
        pd.testing.assert_frame_equal(result, sample_features)


class TestAdvancedFeatureSelector:
    """Tests for advanced feature selection."""

    @pytest.fixture
    def sample_df_with_target(self, sample_features, sample_target):
        """Create DataFrame with features and target column."""
        df = sample_features.copy()
        df["target"] = sample_target.values
        return df

    def test_init(self):
        """Test initialization."""
        selector = AdvancedFeatureSelector(n_features=50)
        assert selector.n_features == 50
        assert selector.method == "combined"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        selector = AdvancedFeatureSelector(
            n_features=100,
            method="correlation",
            correlation_threshold=0.90,
        )
        assert selector.n_features == 100
        assert selector.method == "correlation"

    def test_fit_transform_importance(self, sample_df_with_target):
        """Test feature selection using importance method."""
        selector = AdvancedFeatureSelector(n_features=10, method="importance", verbose=False)

        result, features = selector.fit_transform(sample_df_with_target, "target")

        assert result.shape[1] == 10
        assert len(selector.selected_features) == 10

    def test_fit_transform_correlation(self, sample_df_with_target):
        """Test feature selection using correlation method."""
        selector = AdvancedFeatureSelector(n_features=10, method="correlation", verbose=False)

        result, features = selector.fit_transform(sample_df_with_target, "target")

        assert result.shape[1] == 10
        assert len(selector.selected_features) == 10

    def test_fit_transform_mutual_info(self, sample_df_with_target):
        """Test feature selection using mutual information method."""
        selector = AdvancedFeatureSelector(n_features=10, method="mutual_info", verbose=False)

        result, features = selector.fit_transform(sample_df_with_target, "target")

        assert result.shape[1] == 10
        assert len(selector.selected_features) == 10

    def test_fit_transform_combined(self, sample_df_with_target):
        """Test feature selection using combined method."""
        selector = AdvancedFeatureSelector(n_features=10, method="combined", verbose=False)

        result, features = selector.fit_transform(sample_df_with_target, "target")

        assert result.shape[1] == 10
        assert len(selector.selected_features) == 10
        assert selector.feature_importance is not None

    def test_transform_after_fit(self, sample_df_with_target):
        """Test transform on new data after fitting."""
        selector = AdvancedFeatureSelector(n_features=10, method="combined", verbose=False)

        # Fit on training data
        selector.fit(sample_df_with_target, "target")

        # Transform new data (without target)
        new_data = sample_df_with_target.drop(columns=["target"]).copy()
        result = selector.transform(new_data)

        assert result.shape[1] == 10
        assert list(result.columns) == selector.selected_features

    def test_get_feature_importance(self, sample_df_with_target):
        """Test getting feature importance after fitting."""
        selector = AdvancedFeatureSelector(n_features=10, method="combined", verbose=False)
        selector.fit(sample_df_with_target, "target")

        importance = selector.get_feature_importance()

        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_n_features_greater_than_available(self, sample_df_with_target):
        """Test when n_features exceeds available features."""
        selector = AdvancedFeatureSelector(n_features=100, verbose=False)

        result, features = selector.fit_transform(sample_df_with_target, "target")

        # Should return all available features (20 minus target)
        assert result.shape[1] <= 20


class TestDataPipeline:
    """Tests for complete data pipeline."""

    def test_init(self):
        """Test initialization."""
        pipeline = DataPipeline(nan_threshold=0.25, n_features=100, verbose=False)
        assert pipeline.nan_threshold == 0.25
        assert pipeline.n_features == 100

    def test_fit_transform(self, sample_train_data):
        """Test full pipeline fit_transform."""
        pipeline = DataPipeline(
            nan_threshold=0.30,
            n_features=10,
            use_catboost_imputation=False,  # Faster for testing
            verbose=False,
        )

        # Add some NaN values
        df = sample_train_data.copy()
        df.loc[0:10, "D1"] = np.nan

        # Fit and transform (pass target column name)
        result, features = pipeline.fit_transform(df, "market_forward_excess_returns")

        assert not result.isna().any().any()
        assert result.shape[1] <= 10

    def test_transform_test_data(self, sample_train_data):
        """Test transforming test data after fitting on train."""
        pipeline = DataPipeline(
            nan_threshold=0.30,
            n_features=10,
            use_catboost_imputation=False,
            verbose=False,
        )

        # Split data
        train_df = sample_train_data.iloc[:800].copy()
        test_df = sample_train_data.iloc[800:].copy()

        # Fit on training data
        pipeline.fit(train_df, "market_forward_excess_returns")

        # Transform test data
        result, features = pipeline.transform(test_df)

        assert not result.isna().any().any()
        assert result.shape[1] == pipeline.n_features

    def test_get_selected_features(self, sample_train_data):
        """Test getting selected feature names."""
        pipeline = DataPipeline(n_features=10, use_catboost_imputation=False, verbose=False)

        pipeline.fit(sample_train_data, "market_forward_excess_returns")
        selected = pipeline.get_selected_features()

        assert isinstance(selected, list)
        assert len(selected) == 10

    def test_pipeline_with_catboost_imputation(self, sample_features, sample_target):
        """Test pipeline with CatBoost imputation enabled."""
        # Create DataFrame with target column
        df = sample_features.copy()
        df["target"] = sample_target.values

        # Add missing values
        df.loc[0:5, df.columns[0]] = np.nan
        df.loc[10:15, df.columns[1]] = np.nan

        pipeline = DataPipeline(
            nan_threshold=0.30,
            n_features=10,
            use_catboost_imputation=True,
            catboost_iterations=10,
            verbose=False,
        )

        result, features = pipeline.fit_transform(df, "target")

        assert not result.isna().any().any()
        assert result.shape[1] == 10
