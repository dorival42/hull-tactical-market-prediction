"""Kaggle data loader for Hull Tactical competition."""

import zipfile
from pathlib import Path

import pandas as pd

from src.utils.config import Config, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KaggleDataLoader:
    """
    Load data from Kaggle API or local cache.

    This class handles downloading competition data from Kaggle,
    caching it locally, and loading it into pandas DataFrames.
    """

    def __init__(
        self,
        competition: str | None = None,
        cache_dir: str | None = None,
    ):
        """
        Initialize Kaggle data loader.

        Args:
            competition: Kaggle competition name.
            cache_dir: Directory for caching downloaded data.
        """
        config = Config()
        self.competition = competition or config.get(
            "data.kaggle_dataset", "hull-tactical/hull-tactical-index-tracking"
        )
        self.cache_dir = Path(
            cache_dir or config.get("data.cache_dir", "artifacts/data")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._train_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None

    def _get_kaggle_api(self):
        """Get authenticated Kaggle API client."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            return api
        except Exception as e:
            logger.error(f"Failed to authenticate Kaggle API: {e}")
            raise RuntimeError(
                "Kaggle API authentication failed. "
                "Make sure KAGGLE_USERNAME and KAGGLE_KEY are set, "
                "or ~/.kaggle/kaggle.json exists."
            ) from e

    def download_data(self, force: bool = False) -> Path:
        """
        Download competition data from Kaggle.

        Args:
            force: Force re-download even if cached.

        Returns:
            Path to the downloaded data directory.
        """
        train_path = self.cache_dir / "train.csv"
        test_path = self.cache_dir / "test.csv"

        if not force and train_path.exists() and test_path.exists():
            logger.info(f"Using cached data from {self.cache_dir}")
            return self.cache_dir

        logger.info(f"Downloading data from Kaggle: {self.competition}")

        api = self._get_kaggle_api()

        # Download competition files
        api.competition_download_files(
            competition=self.competition.split("/")[-1],
            path=str(self.cache_dir),
            quiet=False,
        )

        # Extract if zip file exists
        zip_path = self.cache_dir / f"{self.competition.split('/')[-1]}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.cache_dir)
            zip_path.unlink()
            logger.info(f"Extracted data to {self.cache_dir}")

        return self.cache_dir

    def load_train(
        self,
        download: bool = True,
        cutoff_date_id: int | None = None,
    ) -> pd.DataFrame:
        """
        Load training data.

        Args:
            download: Whether to download if not cached.
            cutoff_date_id: Optional cutoff for filtering recent data.

        Returns:
            Training DataFrame.
        """
        if self._train_df is not None and cutoff_date_id is None:
            return self._train_df

        train_path = self.cache_dir / "train.csv"

        # Try local project root first
        if not train_path.exists():
            project_root = get_project_root()
            local_train = project_root / "train.csv"
            if local_train.exists():
                train_path = local_train
            elif download:
                self.download_data()

        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                "Run download_data() or set KAGGLE credentials."
            )

        logger.info(f"Loading training data from {train_path}")
        df = pd.read_csv(train_path)

        if cutoff_date_id is not None:
            df = df[df["date_id"] >= cutoff_date_id].copy()
            logger.info(f"Filtered to date_id >= {cutoff_date_id}: {len(df)} rows")

        if cutoff_date_id is None:
            self._train_df = df

        return df

    def load_test(self, download: bool = True) -> pd.DataFrame:
        """
        Load test data.

        Args:
            download: Whether to download if not cached.

        Returns:
            Test DataFrame.
        """
        if self._test_df is not None:
            return self._test_df

        test_path = self.cache_dir / "test.csv"

        # Try local project root first
        if not test_path.exists():
            project_root = get_project_root()
            local_test = project_root / "test.csv"
            if local_test.exists():
                test_path = local_test
            elif download:
                self.download_data()

        if not test_path.exists():
            raise FileNotFoundError(
                f"Test data not found at {test_path}. "
                "Run download_data() or set KAGGLE credentials."
            )

        logger.info(f"Loading test data from {test_path}")
        self._test_df = pd.read_csv(test_path)

        return self._test_df

    def load_both(
        self,
        download: bool = True,
        cutoff_date_id: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both train and test data.

        Args:
            download: Whether to download if not cached.
            cutoff_date_id: Optional cutoff for filtering training data.

        Returns:
            Tuple of (train_df, test_df).
        """
        train_df = self.load_train(download=download, cutoff_date_id=cutoff_date_id)
        test_df = self.load_test(download=download)
        return train_df, test_df

    def get_data_info(self) -> dict[str, any]:
        """
        Get information about the loaded data.

        Returns:
            Dictionary with data statistics.
        """
        info = {
            "competition": self.competition,
            "cache_dir": str(self.cache_dir),
        }

        train_path = self.cache_dir / "train.csv"
        test_path = self.cache_dir / "test.csv"

        if train_path.exists():
            train_df = self.load_train(download=False)
            info["train"] = {
                "rows": len(train_df),
                "columns": len(train_df.columns),
                "date_id_range": (
                    int(train_df["date_id"].min()),
                    int(train_df["date_id"].max()),
                ),
                "memory_mb": train_df.memory_usage(deep=True).sum() / 1024 / 1024,
            }

        if test_path.exists():
            test_df = self.load_test(download=False)
            info["test"] = {
                "rows": len(test_df),
                "columns": len(test_df.columns),
            }

        return info

    def clear_cache(self) -> None:
        """Clear the data cache."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared cache: {self.cache_dir}")

        self._train_df = None
        self._test_df = None
