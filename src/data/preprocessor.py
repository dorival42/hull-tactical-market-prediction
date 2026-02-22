"""
Advanced Data Preprocessing Pipeline for Hull Tactical.

Features:
- Column filtering based on missing value threshold (30%)
- CatBoost-based missing value imputation
- Advanced feature selection
- No date cutoff (uses all available data)
"""


import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CatBoostImputer:
    """
    CatBoost-based imputer for missing values.

    Uses CatBoost regression to predict missing values based on
    other features, providing more accurate imputation than
    simple statistical methods.
    """

    def __init__(
        self,
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 4,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize CatBoost imputer.

        Args:
            iterations: Number of boosting iterations.
            learning_rate: Learning rate.
            depth: Tree depth.
            random_state: Random seed.
            verbose: Whether to show training progress.
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.verbose = verbose
        self.models: dict[str, any] = {}
        self.column_order: list[str] = []
        self._fitted = False

    def _get_catboost_model(self):
        """Create a CatBoost regressor instance."""
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                random_seed=self.random_state,
                verbose=self.verbose,
                allow_writing_files=False,
            )
        except ImportError as err:
            raise ImportError(
                "CatBoost is required for CatBoostImputer. "
                "Install with: pip install catboost"
            ) from err

    def fit(self, X: pd.DataFrame) -> "CatBoostImputer":
        """
        Fit the imputer on training data.

        For each column with missing values, trains a CatBoost model
        to predict the missing values from other columns.

        Args:
            X: Training DataFrame.

        Returns:
            Self for chaining.
        """
        logger.info("Fitting CatBoost imputer...")

        X = X.copy()
        self.column_order = X.columns.tolist()

        # Identify columns with missing values
        cols_with_nan = X.columns[X.isna().any()].tolist()

        if not cols_with_nan:
            logger.info("No missing values found, nothing to impute")
            self._fitted = True
            return self

        logger.info(f"Training imputers for {len(cols_with_nan)} columns with missing values")

        # Sort columns by number of missing values (impute least missing first)
        nan_counts = X[cols_with_nan].isna().sum().sort_values()
        cols_with_nan = nan_counts.index.tolist()

        # Iteratively fit models
        X_imputed = X.copy()

        for col in cols_with_nan:
            # Get rows without missing values in this column
            mask_not_nan = ~X_imputed[col].isna()

            if mask_not_nan.sum() < 50:
                logger.warning(f"Not enough data to train imputer for {col}, using median")
                self.models[col] = {"type": "median", "value": X_imputed[col].median()}
                X_imputed[col] = X_imputed[col].fillna(self.models[col]["value"])
                continue

            # Features for this column (all other columns)
            feature_cols = [c for c in X_imputed.columns if c != col]

            # Prepare training data
            X_train = X_imputed.loc[mask_not_nan, feature_cols].fillna(0)
            y_train = X_imputed.loc[mask_not_nan, col]

            # Train CatBoost model
            model = self._get_catboost_model()
            model.fit(X_train, y_train, verbose=False)

            self.models[col] = {"type": "catboost", "model": model, "features": feature_cols}

            # Impute missing values for subsequent columns
            mask_nan = X_imputed[col].isna()
            if mask_nan.any():
                X_pred = X_imputed.loc[mask_nan, feature_cols].fillna(0)
                X_imputed.loc[mask_nan, col] = model.predict(X_pred)

        self._fitted = True
        logger.info(f"CatBoost imputer fitted for {len(self.models)} columns")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by imputing missing values.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with imputed values.
        """
        if not self._fitted:
            raise ValueError("Imputer not fitted. Call fit() first.")

        X = X.copy()

        # Ensure column order matches training
        for col in self.column_order:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self.column_order]

        # Apply imputation
        for col, model_info in self.models.items():
            if col not in X.columns:
                continue

            mask_nan = X[col].isna()
            if not mask_nan.any():
                continue

            if model_info["type"] == "median":
                X.loc[mask_nan, col] = model_info["value"]
            else:
                feature_cols = model_info["features"]
                X_pred = X.loc[mask_nan, feature_cols].fillna(0)
                X.loc[mask_nan, col] = model_info["model"].predict(X_pred)

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class DataPreprocessor:
    """
    Advanced data preprocessing pipeline.

    Features:
    - Remove columns with >30% missing values
    - CatBoost-based imputation for remaining missing values
    - Handle infinite values
    - No date cutoff (uses all data)
    """

    def __init__(
        self,
        nan_threshold: float = 0.30,
        use_catboost_imputation: bool = True,
        catboost_iterations: int = 100,
        verbose: bool = True,
    ):
        """
        Initialize preprocessor.

        Args:
            nan_threshold: Drop columns with more than this fraction of NaN.
            use_catboost_imputation: Whether to use CatBoost for imputation.
            catboost_iterations: Number of iterations for CatBoost imputer.
            verbose: Whether to print progress messages.
        """
        self.nan_threshold = nan_threshold
        self.use_catboost_imputation = use_catboost_imputation
        self.catboost_iterations = catboost_iterations
        self.verbose = verbose

        self._dropped_columns: list[str] = []
        self._kept_columns: list[str] = []
        self._imputer: CatBoostImputer | None = None
        self._median_values: dict[str, float] = {}
        self._fitted = False

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data.

        Steps:
        1. Identify and mark columns with >30% missing values for removal
        2. Fit CatBoost imputer on remaining columns
        3. Store statistics for transform

        Args:
            df: Training DataFrame.

        Returns:
            Self for chaining.
        """
        self._log("=" * 60)
        self._log("DATA PREPROCESSING - FIT")
        self._log("=" * 60)

        df = df.copy()

        # Step 1: Identify columns to drop (>30% NaN)
        self._log(f"\n1. Analyzing missing values (threshold: {self.nan_threshold*100:.0f}%)")

        nan_ratios = df.isna().mean()
        self._dropped_columns = nan_ratios[nan_ratios > self.nan_threshold].index.tolist()
        self._kept_columns = [c for c in df.columns if c not in self._dropped_columns]

        self._log(f"   - Total columns: {len(df.columns)}")
        self._log(f"   - Columns to drop (>{self.nan_threshold*100:.0f}% NaN): {len(self._dropped_columns)}")
        self._log(f"   - Columns to keep: {len(self._kept_columns)}")

        if self._dropped_columns:
            self._log("\n   Dropped columns:")
            for col in self._dropped_columns[:10]:
                pct = nan_ratios[col] * 100
                self._log(f"     - {col}: {pct:.1f}% missing")
            if len(self._dropped_columns) > 10:
                self._log(f"     ... and {len(self._dropped_columns) - 10} more")

        # Keep only selected columns
        df_filtered = df[self._kept_columns].copy()

        # Step 2: Handle infinite values
        self._log("\n2. Handling infinite values")
        numeric_cols = df_filtered.select_dtypes(include=["float64", "float32", "int64"]).columns
        inf_counts = np.isinf(df_filtered[numeric_cols]).sum().sum()
        self._log(f"   - Infinite values found: {inf_counts}")
        df_filtered[numeric_cols] = df_filtered[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Step 3: Fit imputer
        self._log("\n3. Fitting imputer")

        # Store median values as fallback
        for col in numeric_cols:
            self._median_values[col] = df_filtered[col].median()

        if self.use_catboost_imputation:
            self._log("   Using CatBoost imputation")
            self._imputer = CatBoostImputer(
                iterations=self.catboost_iterations,
                verbose=False,
            )
            # Only fit on numeric columns with missing values
            cols_to_impute = df_filtered[numeric_cols].columns[
                df_filtered[numeric_cols].isna().any()
            ].tolist()

            if cols_to_impute:
                self._log(f"   - Columns to impute: {len(cols_to_impute)}")
                self._imputer.fit(df_filtered[numeric_cols])
        else:
            self._log("   Using median imputation")

        self._fitted = True
        self._log("\n" + "=" * 60)
        self._log("PREPROCESSING FIT COMPLETE")
        self._log("=" * 60)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        if not self._fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        self._log("\nTransforming data...")

        df = df.copy()

        # Step 1: Drop high-NaN columns
        cols_to_keep = [c for c in self._kept_columns if c in df.columns]
        df = df[cols_to_keep]

        # Add missing columns with NaN
        for col in self._kept_columns:
            if col not in df.columns:
                df[col] = np.nan

        df = df[self._kept_columns]

        # Step 2: Handle infinite values
        numeric_cols = df.select_dtypes(include=["float64", "float32", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Step 3: Impute missing values
        if self.use_catboost_imputation and self._imputer is not None and self._imputer._fitted:
            df[numeric_cols] = self._imputer.transform(df[numeric_cols])

        # Step 4: Final median imputation for any remaining NaN
        for col in numeric_cols:
            if df[col].isna().any():
                fill_value = self._median_values.get(col, 0)
                df[col] = df[col].fillna(fill_value)

        self._log(f"   - Output shape: {df.shape}")

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_dropped_columns(self) -> list[str]:
        """Get list of dropped columns."""
        return self._dropped_columns.copy()

    def get_kept_columns(self) -> list[str]:
        """Get list of kept columns."""
        return self._kept_columns.copy()

    def get_preprocessing_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a preprocessing report.

        Args:
            df: Original DataFrame.

        Returns:
            DataFrame with column statistics.
        """
        report_data = []

        for col in df.columns:
            nan_count = df[col].isna().sum()
            nan_pct = nan_count / len(df) * 100

            report_data.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": nan_count,
                "missing_pct": nan_pct,
                "will_drop": nan_pct > self.nan_threshold * 100,
                "unique_values": df[col].nunique(),
            })

        report = pd.DataFrame(report_data)
        report = report.sort_values("missing_pct", ascending=False)

        return report


class AdvancedFeatureSelector:
    """
    Advanced feature selection using multiple methods.

    Combines:
    - CatBoost feature importance
    - Correlation analysis
    - Mutual information
    - Recursive feature elimination
    """

    def __init__(
        self,
        n_features: int = 150,
        method: str = "combined",
        correlation_threshold: float = 0.95,
        verbose: bool = True,
    ):
        """
        Initialize feature selector.

        Args:
            n_features: Number of features to select.
            method: Selection method ('catboost', 'correlation', 'mutual_info', 'combined').
            correlation_threshold: Threshold for removing highly correlated features.
            verbose: Whether to print progress.
        """
        self.n_features = n_features
        self.method = method
        self.correlation_threshold = correlation_threshold
        self.verbose = verbose

        self.selected_features: list[str] = []
        self.feature_importance: pd.DataFrame | None = None
        self._fitted = False

    def _log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            logger.info(message)

    def _remove_highly_correlated(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> list[str]:
        """Remove highly correlated features."""
        if len(feature_cols) < 2:
            return feature_cols

        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()

        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find columns to drop
        to_drop = set()
        for col in upper_tri.columns:
            if any(upper_tri[col] > self.correlation_threshold):
                to_drop.add(col)

        remaining = [c for c in feature_cols if c not in to_drop]

        if to_drop:
            self._log(f"   Removed {len(to_drop)} highly correlated features")

        return remaining

    def _catboost_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate feature importance using CatBoost."""
        try:
            from catboost import CatBoostRegressor
        except ImportError as err:
            raise ImportError("CatBoost required for feature selection") from err

        model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=5,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )

        model.fit(X, y, verbose=False)

        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return importance

    def _correlation_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate feature importance based on correlation with target."""
        correlations = []

        for col in X.columns:
            valid_mask = ~(X[col].isna() | pd.isna(y))
            if valid_mask.sum() > 50:
                corr = abs(np.corrcoef(X.loc[valid_mask, col], y[valid_mask])[0, 1])
                correlations.append({"feature": col, "importance": corr})

        importance = pd.DataFrame(correlations)
        importance = importance.sort_values("importance", ascending=False)

        return importance

    def _mutual_info_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate feature importance using mutual information."""
        from sklearn.feature_selection import mutual_info_regression

        X_filled = X.fillna(0)
        mi_scores = mutual_info_regression(X_filled, y, random_state=42)

        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": mi_scores,
        }).sort_values("importance", ascending=False)

        return importance

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        exclude_cols: list[str] | None = None,
    ) -> "AdvancedFeatureSelector":
        """
        Fit feature selector.

        Args:
            df: Training DataFrame.
            target_col: Target column name.
            exclude_cols: Columns to exclude from selection.

        Returns:
            Self for chaining.
        """
        self._log("=" * 60)
        self._log("FEATURE SELECTION")
        self._log("=" * 60)

        # Default excluded columns
        default_exclude = [
            "date_id",
            target_col,
            "forward_returns",
            "risk_free_rate",
            "market_forward_excess_returns",
            "is_scored",
        ]

        if exclude_cols:
            default_exclude.extend(exclude_cols)

        exclude_set = set(default_exclude)

        # Get candidate features
        candidate_cols = [c for c in df.columns if c not in exclude_set]
        self._log(f"\nCandidate features: {len(candidate_cols)}")

        # Prepare data
        X = df[candidate_cols].fillna(0)
        y = df[target_col].values

        # Remove rows with NaN target
        valid_mask = ~pd.isna(y)
        X = X[valid_mask]
        y = y[valid_mask]

        self._log(f"Training samples: {len(X)}")

        # Step 1: Remove highly correlated features
        self._log("\n1. Removing highly correlated features...")
        candidate_cols = self._remove_highly_correlated(X, candidate_cols)
        X = X[candidate_cols]
        self._log(f"   Remaining features: {len(candidate_cols)}")

        # Step 2: Calculate feature importance
        self._log(f"\n2. Calculating feature importance (method: {self.method})")

        if self.method in ("catboost", "importance"):
            importance = self._catboost_importance(X, y)
        elif self.method == "correlation":
            importance = self._correlation_importance(X, y)
        elif self.method == "mutual_info":
            importance = self._mutual_info_importance(X, y)
        elif self.method == "combined":
            # Combine multiple methods
            self._log("   Computing CatBoost importance...")
            imp_catboost = self._catboost_importance(X, y)
            imp_catboost = imp_catboost.rename(columns={"importance": "catboost"})

            self._log("   Computing correlation importance...")
            imp_corr = self._correlation_importance(X, y)
            imp_corr = imp_corr.rename(columns={"importance": "correlation"})

            self._log("   Computing mutual information...")
            imp_mi = self._mutual_info_importance(X, y)
            imp_mi = imp_mi.rename(columns={"importance": "mutual_info"})

            # Merge and normalize
            importance = imp_catboost.merge(imp_corr, on="feature", how="outer")
            importance = importance.merge(imp_mi, on="feature", how="outer")

            # Normalize each score to [0, 1]
            for col in ["catboost", "correlation", "mutual_info"]:
                if col in importance.columns:
                    min_val = importance[col].min()
                    max_val = importance[col].max()
                    if max_val > min_val:
                        importance[col] = (importance[col] - min_val) / (max_val - min_val)
                    else:
                        importance[col] = 0.5

            # Combined score (weighted average)
            importance["importance"] = (
                importance["catboost"] * 0.5 +
                importance["correlation"] * 0.25 +
                importance["mutual_info"] * 0.25
            )

            importance = importance.sort_values("importance", ascending=False)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.feature_importance = importance

        # Step 3: Select top features
        self._log(f"\n3. Selecting top {self.n_features} features")

        # Filter to only available columns
        available_features = [f for f in importance["feature"] if f in candidate_cols]
        self.selected_features = available_features[:self.n_features]

        self._log(f"   Selected {len(self.selected_features)} features")

        # Log top features
        self._log("\n   Top 10 features:")
        for _i, row in importance.head(10).iterrows():
            self._log(f"     {row['feature']}: {row['importance']:.4f}")

        self._fitted = True
        self._log("\n" + "=" * 60)
        self._log("FEATURE SELECTION COMPLETE")
        self._log("=" * 60)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting features.

        Args:
            df: DataFrame to transform.

        Returns:
            DataFrame with selected features only.
        """
        if not self._fitted:
            raise ValueError("Selector not fitted. Call fit() first.")

        # Keep only selected features plus any required columns
        available = [c for c in self.selected_features if c in df.columns]
        return df[available]

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        exclude_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Fit and transform in one step.

        Returns:
            Tuple of (transformed DataFrame, selected feature list).
        """
        self.fit(df, target_col, exclude_cols)
        return self.transform(df), self.selected_features

    def get_selected_features(self) -> list[str]:
        """Get list of selected features."""
        return self.selected_features.copy()

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance DataFrame."""
        if self.feature_importance is None:
            raise ValueError("Selector not fitted")
        return self.feature_importance.copy()

    def save_features(self, filepath: str) -> None:
        """Save selected features to JSON file."""
        import json
        from pathlib import Path

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.selected_features, f, indent=2)

        self._log(f"Saved {len(self.selected_features)} features to {filepath}")

    def load_features(self, filepath: str) -> list[str]:
        """Load selected features from JSON file."""
        import json

        with open(filepath) as f:
            self.selected_features = json.load(f)

        self._fitted = True
        self._log(f"Loaded {len(self.selected_features)} features from {filepath}")

        return self.selected_features


class DataPipeline:
    """
    Complete data preparation pipeline.

    Combines preprocessing and feature selection into a single
    easy-to-use pipeline.
    """

    def __init__(
        self,
        nan_threshold: float = 0.30,
        use_catboost_imputation: bool = True,
        n_features: int = 150,
        feature_selection_method: str = "combined",
        catboost_iterations: int = 100,
        verbose: bool = True,
    ):
        """
        Initialize data pipeline.

        Args:
            nan_threshold: Drop columns with more than this fraction of NaN.
            use_catboost_imputation: Whether to use CatBoost imputation.
            n_features: Number of features to select.
            feature_selection_method: Feature selection method.
            catboost_iterations: Iterations for CatBoost imputer.
            verbose: Whether to print progress.
        """
        self.nan_threshold = nan_threshold
        self.n_features = n_features

        self.preprocessor = DataPreprocessor(
            nan_threshold=nan_threshold,
            use_catboost_imputation=use_catboost_imputation,
            catboost_iterations=catboost_iterations,
            verbose=verbose,
        )

        self.feature_selector = AdvancedFeatureSelector(
            n_features=n_features,
            method=feature_selection_method,
            verbose=verbose,
        )

        self.verbose = verbose
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "market_forward_excess_returns",
    ) -> "DataPipeline":
        """
        Fit the complete pipeline.

        Args:
            df: Training DataFrame.
            target_col: Target column name.

        Returns:
            Self for chaining.
        """
        if self.verbose:
            logger.info("\n" + "=" * 80)
            logger.info("COMPLETE DATA PIPELINE - FIT")
            logger.info("=" * 80)
            logger.info(f"\nInput data: {df.shape[0]} rows x {df.shape[1]} columns")
            logger.info("NOTE: Using ALL data (no date cutoff)")

        # Step 1: Preprocessing
        df_preprocessed = self.preprocessor.fit_transform(df)

        # Step 2: Feature Selection
        self.feature_selector.fit(df_preprocessed, target_col)

        self._fitted = True

        if self.verbose:
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE FIT COMPLETE")
            logger.info(f"Final selected features: {len(self.feature_selector.selected_features)}")
            logger.info("=" * 80)

        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Transform data through the pipeline.

        Args:
            df: DataFrame to transform.

        Returns:
            Tuple of (transformed DataFrame, feature columns).
        """
        if not self._fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        # Apply preprocessing
        df_preprocessed = self.preprocessor.transform(df)

        # Apply feature selection
        df_selected = self.feature_selector.transform(df_preprocessed)

        return df_selected, self.feature_selector.selected_features

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "market_forward_excess_returns",
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Fit and transform in one step.

        Args:
            df: Training DataFrame.
            target_col: Target column name.

        Returns:
            Tuple of (transformed DataFrame, selected features).
        """
        self.fit(df, target_col)
        return self.transform(df)

    def get_selected_features(self) -> list[str]:
        """Get selected features."""
        return self.feature_selector.get_selected_features()

    def save(self, output_dir: str) -> None:
        """
        Save pipeline artifacts.

        Args:
            output_dir: Directory to save artifacts.
        """
        from pathlib import Path

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save feature list
        self.feature_selector.save_features(str(path / "selected_features.json"))

        # Save preprocessing report
        if self.verbose:
            logger.info(f"Pipeline artifacts saved to {output_dir}")
