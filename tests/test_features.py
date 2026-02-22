"""Tests for feature engineering."""

import numpy as np

from src.features.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_init(self):
        """Test initialization."""
        fe = FeatureEngineer(verbose=False)
        assert fe.feature_names is None
        assert fe.numeric_features == []
        assert fe.selected_features == []

    def test_fit(self, sample_train_data):
        """Test fit method."""
        fe = FeatureEngineer(verbose=False)
        fe.fit(sample_train_data)

        assert len(fe.numeric_features) > 0
        assert len(fe._fit_stats) > 0

    def test_transform(self, sample_train_data):
        """Test transform method."""
        fe = FeatureEngineer(verbose=False)
        fe.fit(sample_train_data)

        transformed = fe.transform(sample_train_data)

        # Should have more columns after transformation
        assert transformed.shape[1] > sample_train_data.shape[1]

        # Should not have NaN in result
        numeric_cols = transformed.select_dtypes(include=["float64", "int64"]).columns
        assert not transformed[numeric_cols].isna().any().any()

    def test_fit_transform(self, sample_train_data):
        """Test fit_transform method."""
        fe = FeatureEngineer(verbose=False)
        transformed = fe.fit_transform(sample_train_data)

        assert transformed.shape[0] == sample_train_data.shape[0]
        assert transformed.shape[1] > sample_train_data.shape[1]

    def test_create_lagged_features(self, sample_train_data):
        """Test lagged feature creation."""
        fe = FeatureEngineer(verbose=False)
        transformed = fe._create_lagged_features(sample_train_data.copy())

        assert "lagged_market_forward_excess_returns" in transformed.columns
        assert "lagged_forward_returns" in transformed.columns
        assert "lagged_risk_free_rate" in transformed.columns

    def test_create_target_features(self, sample_train_data):
        """Test target-based feature creation."""
        fe = FeatureEngineer(verbose=False)

        # First create lagged features
        df = fe._create_lagged_features(sample_train_data.copy())
        df = fe._create_target_features(df)

        # Check for expected features
        expected_features = [
            "feat_lagged_target",
            "feat_lagged_target_sign",
            "feat_lagged_target_abs",
            "feat_target_rolling_mean_5",
            "feat_target_zscore",
        ]

        for feat in expected_features:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_create_category_features(self, sample_train_data):
        """Test category-based feature creation."""
        fe = FeatureEngineer(verbose=False)
        df = fe._create_category_features(sample_train_data.copy())

        # Check for category aggregate features
        expected_features = [
            "feat_v_mean",
            "feat_m_mean",
            "feat_s_mean",
            "feat_p_mean",
            "feat_i_mean",
            "feat_e_mean",
        ]

        for feat in expected_features:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_create_time_features(self, sample_train_data):
        """Test time-based feature creation."""
        fe = FeatureEngineer(verbose=False)
        df = fe._create_time_features(sample_train_data.copy())

        expected_features = [
            "feat_day_of_year",
            "feat_week_of_year",
            "feat_day_sin",
            "feat_day_cos",
        ]

        for feat in expected_features:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_handle_missing_values(self, sample_train_data):
        """Test missing value handling."""
        fe = FeatureEngineer(verbose=False)

        # Add some missing values
        df = sample_train_data.copy()
        df.loc[0:10, "D1"] = np.nan
        df.loc[0:5, "E1"] = np.inf

        cleaned = fe._handle_missing_values(df)

        numeric_cols = cleaned.select_dtypes(include=["float64", "int64"]).columns
        assert not cleaned[numeric_cols].isna().any().any()
        assert not np.isinf(cleaned[numeric_cols]).any().any()

    def test_select_features(self, sample_train_data):
        """Test feature selection."""
        fe = FeatureEngineer(verbose=False)
        transformed = fe.fit_transform(sample_train_data)

        # Remove rows with NaN in target
        transformed = transformed.dropna(subset=["market_forward_excess_returns"])

        selected = fe.select_features(
            transformed,
            target_col="market_forward_excess_returns",
            method="lgb_importance",
            n_features=20,
        )

        assert len(selected) == 20
        assert "market_forward_excess_returns" not in selected
        assert "date_id" not in selected

    def test_select_features_correlation(self, sample_train_data):
        """Test feature selection with correlation method."""
        fe = FeatureEngineer(verbose=False)
        transformed = fe.fit_transform(sample_train_data)
        transformed = transformed.dropna(subset=["market_forward_excess_returns"])

        selected = fe.select_features(
            transformed,
            target_col="market_forward_excess_returns",
            method="correlation",
            n_features=20,
        )

        assert len(selected) <= 20

    def test_get_feature_stats(self, sample_train_data):
        """Test feature statistics."""
        fe = FeatureEngineer(verbose=False)
        stats = fe.get_feature_stats(sample_train_data)

        assert "feature" in stats.columns
        assert "missing_pct" in stats.columns
        assert "mean" in stats.columns
        assert "std" in stats.columns
