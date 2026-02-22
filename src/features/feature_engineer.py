"""Feature Engineering for Hull Tactical Market Prediction."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_feature_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for Hull Tactical competition.

    Creates advanced features including:
    - Lagged target features
    - Rolling statistics
    - Technical indicators (RSI, MACD)
    - Category-based aggregations
    - Cross-feature interactions
    - Time-based features
    """

    def __init__(self, verbose: bool = True, config: dict | None = None):
        """
        Initialize feature engineer.

        Args:
            verbose: Whether to print progress messages.
            config: Optional configuration dictionary.
        """
        self.verbose = verbose
        self.config = config or load_feature_config()

        self.feature_names: list[str] | None = None
        self.numeric_features: list[str] = []
        self.selected_features: list[str] = []
        self.scaler: StandardScaler | None = None
        self._fit_stats: dict[str, dict] = {}

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit the feature engineer on training data.

        Args:
            df: Training DataFrame.

        Returns:
            Self for chaining.
        """
        exclude_cols = [
            "date_id",
            "forward_returns",
            "risk_free_rate",
            "market_forward_excess_returns",
            "is_scored",
        ]

        self.numeric_features = [
            col
            for col in df.columns
            if col not in exclude_cols and df[col].dtype in ["float64", "int64"]
        ]

        self._log(f"Identified {len(self.numeric_features)} numeric features")
        self._calculate_imputation_stats(df)

        return self

    def _calculate_imputation_stats(self, df: pd.DataFrame) -> None:
        """Calculate statistics for imputation."""
        self._log("Calculating imputation statistics...")

        for col in self.numeric_features:
            if col in df.columns:
                self._fit_stats[col] = {
                    "median": df[col].median(),
                    "mean": df[col].mean(),
                    "nan_count": df[col].isna().sum(),
                    "nan_pct": df[col].isna().sum() / len(df) * 100,
                }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with feature engineering.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame with new features.
        """
        self._log(f"Transforming data: input shape {df.shape}")
        df = df.copy()

        # Create lagged features if needed
        df = self._create_lagged_features(df)

        # Create all engineered features
        df = self._create_features(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        self._log(f"Transformation complete: output shape {df.shape}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for train set compatibility."""
        if "lagged_market_forward_excess_returns" not in df.columns:
            self._log("Creating lagged features for train set...")

            if "market_forward_excess_returns" in df.columns:
                df["lagged_market_forward_excess_returns"] = df[
                    "market_forward_excess_returns"
                ].shift(1)

            if "forward_returns" in df.columns:
                df["lagged_forward_returns"] = df["forward_returns"].shift(1)

            if "risk_free_rate" in df.columns:
                df["lagged_risk_free_rate"] = df["risk_free_rate"].shift(1)
        else:
            self._log("Lagged features already present (test set)")

        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features."""
        self._log("Creating engineered features...")
        initial_cols = len(df.columns)

        # 1. Target lagged features
        df = self._create_target_features(df)

        # 2. Rolling features for key columns
        df = self._create_rolling_features(df)

        # 3. Category-based features
        df = self._create_category_features(df)

        # 4. Technical indicators
        df = self._create_technical_indicators(df)

        # 5. Interaction features
        df = self._create_interaction_features(df)

        # 6. Time-based features
        df = self._create_time_features(df)

        final_cols = len(df.columns)
        self._log(f"Created {final_cols - initial_cols} new features")

        return df

    def _create_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on lagged target."""
        target_col = "lagged_market_forward_excess_returns"

        if target_col not in df.columns:
            return df

        self._log("Creating target-based features...")

        # Basic lagged target
        df["feat_lagged_target"] = df[target_col]
        df["feat_lagged_target_sign"] = (df[target_col] > 0).astype(int)
        df["feat_lagged_target_abs"] = df[target_col].abs()

        # Multiple lags
        for lag in [1, 2]:
            df[f"feat_lagged_target_{lag + 1}"] = df[target_col].shift(lag)

        # Rolling statistics
        for window in [5, 20]:
            df[f"feat_target_rolling_mean_{window}"] = (
                df[target_col].rolling(window, min_periods=1).mean()
            )
            df[f"feat_target_rolling_std_{window}"] = (
                df[target_col].rolling(window, min_periods=1).std()
            )

        # Volatility features
        df["feat_target_volatility_5"] = df[target_col].rolling(5, min_periods=1).std()
        df["feat_target_volatility_20"] = df[target_col].rolling(20, min_periods=1).std()

        # Mean reversion
        rolling_mean = df[target_col].rolling(20, min_periods=1).mean()
        df["feat_mean_reversion"] = df[target_col] - rolling_mean

        # Autocorrelation
        df["feat_autocorr"] = df[target_col] * df[target_col].shift(1)

        # Momentum
        df["feat_momentum_2days"] = df[target_col] * df[target_col].shift(1)

        # Z-score
        rolling_std = df[target_col].rolling(20, min_periods=1).std()
        df["feat_target_zscore"] = (df[target_col] - rolling_mean) / (rolling_std + 1e-8)

        # Additional lagged features
        if "lagged_forward_returns" in df.columns:
            df["feat_lagged_returns"] = df["lagged_forward_returns"]

        if "lagged_risk_free_rate" in df.columns:
            df["feat_lagged_rfr"] = df["lagged_risk_free_rate"]

        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistics for key columns."""
        target_col = "lagged_market_forward_excess_returns"
        key_cols = ["lagged_forward_returns", "lagged_risk_free_rate", target_col]

        for col in key_cols:
            if col not in df.columns:
                continue

            self._log(f"Creating rolling features for {col}")

            for window in [5, 10, 20, 60]:
                # Rolling mean
                df[f"feat_{col}_rolling_mean_{window}d"] = (
                    df[col].rolling(window, min_periods=1).mean()
                )
                # Rolling std
                df[f"feat_{col}_rolling_std_{window}d"] = (
                    df[col].rolling(window, min_periods=1).std()
                )
                # Rolling max
                df[f"feat_{col}_rolling_max_{window}d"] = (
                    df[col].rolling(window, min_periods=1).max()
                )
                # Rolling min
                df[f"feat_{col}_rolling_min_{window}d"] = (
                    df[col].rolling(window, min_periods=1).min()
                )
                # Rolling skew
                df[f"feat_{col}_rolling_skew_{window}d"] = (
                    df[col].rolling(window, min_periods=1).skew()
                )
                # Rolling kurtosis
                df[f"feat_{col}_rolling_kurtosis_{window}d"] = (
                    df[col].rolling(window, min_periods=1).kurt()
                )

        return df

    def _create_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on column categories (V*, M*, S*, etc.)."""
        categories = {
            "v": "Volatility",
            "m": "Momentum",
            "s": "Sentiment",
            "p": "Price",
            "i": "Interest",
            "e": "Economic",
        }

        for prefix, name in categories.items():
            cols = [
                c
                for c in df.columns
                if c.startswith(prefix.upper())
                and len(c) > 1
                and c[1:].replace(".", "").isdigit()
            ]

            if not cols:
                continue

            self._log(f"Creating {name} features from {len(cols)} columns")

            # Basic statistics
            df[f"feat_{prefix}_mean"] = df[cols].mean(axis=1)
            df[f"feat_{prefix}_std"] = df[cols].std(axis=1)
            df[f"feat_{prefix}_max"] = df[cols].max(axis=1)
            df[f"feat_{prefix}_min"] = df[cols].min(axis=1)

            # Category-specific features
            if prefix == "v":  # Volatility
                df["feat_v_range"] = df["feat_v_max"] - df["feat_v_min"]
                df["feat_v_median"] = df[cols].median(axis=1)
                df["feat_high_vol"] = (
                    df["feat_v_mean"] > df["feat_v_mean"].quantile(0.75)
                ).astype(int)
                df["feat_low_vol"] = (
                    df["feat_v_mean"] < df["feat_v_mean"].quantile(0.25)
                ).astype(int)
                df["feat_v_percentile"] = (
                    df["feat_v_mean"].rolling(252, min_periods=1).rank(pct=True)
                )
                df["feat_v_change"] = df["feat_v_mean"] - df["feat_v_mean"].shift(5)

            elif prefix == "m":  # Momentum
                df["feat_m_sum"] = df[cols].sum(axis=1)
                df["feat_momentum_strength"] = df[cols].abs().sum(axis=1)
                df["feat_positive_momentum"] = (df[cols] > 0).sum(axis=1)
                df["feat_negative_momentum"] = (df[cols] < 0).sum(axis=1)
                df["feat_momentum_balance"] = (
                    df["feat_positive_momentum"] - df["feat_negative_momentum"]
                )
                df["feat_momentum_consistency"] = df["feat_momentum_balance"] / (
                    len(cols) + 1e-8
                )
                df["feat_m_change"] = df["feat_m_mean"] - df["feat_m_mean"].shift(5)

            elif prefix == "s":  # Sentiment
                df["feat_positive_sentiment"] = (df["feat_s_mean"] > 0).astype(int)
                df["feat_extreme_sentiment"] = (
                    df["feat_s_mean"].abs() > df["feat_s_mean"].abs().quantile(0.9)
                ).astype(int)
                df["feat_s_change"] = df["feat_s_mean"] - df["feat_s_mean"].shift(5)

            elif prefix == "p":  # Price
                df["feat_p_change"] = df["feat_p_mean"] - df["feat_p_mean"].shift(5)

            elif prefix == "i":  # Interest rates
                df["feat_i_spread"] = df["feat_i_max"] - df["feat_i_min"]
                if len(cols) >= 3:
                    df["feat_i_slope"] = df[cols[-1]] - df[cols[0]]
                df["feat_i_change"] = df["feat_i_mean"] - df["feat_i_mean"].shift(20)

            elif prefix == "e":  # Economic
                df["feat_e_change"] = df["feat_e_mean"] - df["feat_e_mean"].shift(20)

        return df

    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create RSI and MACD indicators."""
        target_col = "lagged_market_forward_excess_returns"
        key_cols = ["lagged_forward_returns", target_col]

        def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
            """Calculate RSI indicator."""
            delta = data.diff()
            gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))

        for col in key_cols:
            if col not in df.columns:
                continue

            self._log(f"Creating technical indicators for {col}")

            # RSI
            for window in [7, 14, 21]:
                df[f"feat_{col}_rsi_{window}"] = calculate_rsi(df[col], window)

            # MACD
            ema_12 = df[col].ewm(span=12, adjust=False).mean()
            ema_26 = df[col].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()

            df[f"feat_{col}_macd"] = macd
            df[f"feat_{col}_macd_signal"] = signal
            df[f"feat_{col}_macd_hist"] = macd - signal

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between categories."""
        self._log("Creating interaction features...")

        # Volatility x Momentum
        if "feat_v_mean" in df.columns and "feat_m_mean" in df.columns:
            df["feat_vol_momentum_interaction"] = df["feat_v_mean"] * df["feat_m_mean"]
            df["feat_vol_momentum_ratio"] = df["feat_v_mean"] / (
                df["feat_m_mean"].abs() + 1e-8
            )

        # Sentiment x Volatility
        if "feat_s_mean" in df.columns and "feat_v_mean" in df.columns:
            df["feat_sentiment_vol_interaction"] = df["feat_s_mean"] * df["feat_v_mean"]

        # Momentum x Target
        if "feat_m_mean" in df.columns and "feat_lagged_target" in df.columns:
            df["feat_momentum_target_alignment"] = (
                np.sign(df["feat_m_mean"]) == np.sign(df["feat_lagged_target"])
            ).astype(int)
            df["feat_momentum_divergence"] = (
                np.sign(df["feat_m_mean"]) != np.sign(df["feat_lagged_target"])
            ).astype(int)

        # Price x Volatility
        if "feat_p_mean" in df.columns and "feat_v_mean" in df.columns:
            df["feat_price_vol_ratio"] = df["feat_p_mean"] / (df["feat_v_mean"] + 1e-8)

        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based periodic features."""
        self._log("Creating time features...")

        # Approximate day/week/month based on date_id
        # Assuming 252 trading days per year
        df["feat_day_of_year"] = df["date_id"] % 252
        df["feat_week_of_year"] = (df["date_id"] % 252) // 5
        df["feat_month_of_year"] = (df["date_id"] % 252) // 21

        # Cyclical encoding
        df["feat_day_sin"] = np.sin(2 * np.pi * df["feat_day_of_year"] / 252)
        df["feat_day_cos"] = np.cos(2 * np.pi * df["feat_day_of_year"] / 252)
        df["feat_month_sin"] = np.sin(2 * np.pi * df["feat_month_of_year"] / 12)
        df["feat_month_cos"] = np.cos(2 * np.pi * df["feat_month_of_year"] / 12)

        return df

    def _handle_missing_values(
        self, df: pd.DataFrame, nan_threshold: float = 0.30
    ) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        # Drop columns with too many NaN
        nan_ratio = df.isna().mean()
        cols_to_drop = nan_ratio[nan_ratio > nan_threshold].index
        df = df.drop(columns=cols_to_drop)

        # Get numeric columns
        num_cols = df.select_dtypes(include=["float64", "float32", "int64"]).columns

        # Replace inf values
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

        # Impute with median
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        return df

    def select_features(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        method: str = "lgb_importance",
        n_features: int = 150,
    ) -> list[str]:
        """
        Select top features using specified method.

        Args:
            df_train: Training DataFrame.
            target_col: Target column name.
            method: Selection method ('lgb_importance' or 'correlation').
            n_features: Number of features to select.

        Returns:
            List of selected feature names.
        """
        exclude_cols = [
            "date_id",
            target_col,
            "forward_returns",
            "risk_free_rate",
            "market_forward_excess_returns",
            "is_scored",
        ]

        candidate_cols = [col for col in df_train.columns if col not in exclude_cols]
        X = df_train[candidate_cols].fillna(0)
        y = df_train[target_col].values

        if method == "lgb_importance":
            model = lgb.LGBMRegressor(
                random_state=42,
                n_estimators=300,
                learning_rate=0.05,
                verbosity=-1,
            )
            model.fit(X, y)

            importance_df = pd.DataFrame(
                {"feature": candidate_cols, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            selected = importance_df.head(n_features)["feature"].tolist()

        elif method == "correlation":
            correlations = []
            for col in candidate_cols:
                subset = df_train[[col, target_col]].dropna()
                if len(subset) > 50:
                    corr = abs(subset[col].corr(subset[target_col]))
                    correlations.append((col, corr))

            correlations.sort(key=lambda x: x[1], reverse=True)
            selected = [c for c, _ in correlations[:n_features]]

        else:
            raise ValueError(f"Unknown method: {method}")

        self._log(f"Selected {len(selected)} features using {method}")
        self.selected_features = selected

        return selected

    def save_selected_features(self, filepath: str) -> None:
        """Save selected features to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.selected_features, f, indent=2)

        self._log(f"Saved {len(self.selected_features)} features to {filepath}")

    def load_selected_features(self, filepath: str) -> list[str]:
        """Load selected features from JSON file."""
        with open(filepath) as f:
            self.selected_features = json.load(f)

        self._log(f"Loaded {len(self.selected_features)} features from {filepath}")
        return self.selected_features

    def normalize_features(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize features using StandardScaler.

        Args:
            df_train: Training DataFrame.
            df_test: Test DataFrame.
            feature_cols: Columns to normalize.

        Returns:
            Tuple of normalized (train, test) DataFrames.
        """
        self._log(f"Normalizing {len(feature_cols)} features...")

        self.scaler = StandardScaler()

        df_train = df_train.copy()
        df_test = df_test.copy()

        df_train[feature_cols] = self.scaler.fit_transform(
            df_train[feature_cols].fillna(0)
        )
        df_test[feature_cols] = self.scaler.transform(df_test[feature_cols].fillna(0))

        return df_train, df_test

    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics for all features."""
        stats = []

        for col in df.columns:
            if col != "date_id":
                stats.append(
                    {
                        "feature": col,
                        "missing_pct": df[col].isna().sum() / len(df) * 100,
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max(),
                    }
                )

        return pd.DataFrame(stats)
