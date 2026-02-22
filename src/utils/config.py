"""Configuration management for Hull Tactical."""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_config(
    config_path: str | None = None,
    config_name: str = "config.yaml",
    load_env: bool = True,
) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config directory. If None, uses default.
        config_name: Name of the config file.
        load_env: Whether to load environment variables from .env file.

    Returns:
        OmegaConf DictConfig object with configuration.
    """
    if load_env:
        load_dotenv()

    if config_path is None:
        config_path = get_project_root() / "configs"
    else:
        config_path = Path(config_path)

    config_file = config_path / config_name
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config = OmegaConf.load(config_file)

    # Resolve environment variables
    OmegaConf.resolve(config)

    return config


def load_model_params(config_path: str | None = None) -> DictConfig:
    """Load model parameters configuration."""
    return load_config(config_path, "model_params.yaml")


def load_feature_config(config_path: str | None = None) -> DictConfig:
    """Load feature engineering configuration."""
    return load_config(config_path, "features.yaml")


class Config:
    """Configuration manager class."""

    _instance: Optional["Config"] = None
    _config: DictConfig | None = None
    _model_params: DictConfig | None = None
    _feature_config: DictConfig | None = None

    def __new__(cls) -> "Config":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration."""
        if self._config is None:
            self.reload()

    def reload(self) -> None:
        """Reload all configurations."""
        self._config = load_config()
        self._model_params = load_model_params()
        self._feature_config = load_feature_config()

    @property
    def config(self) -> DictConfig:
        """Get main configuration."""
        if self._config is None:
            self.reload()
        return self._config

    @property
    def model_params(self) -> DictConfig:
        """Get model parameters."""
        if self._model_params is None:
            self._model_params = load_model_params()
        return self._model_params

    @property
    def feature_config(self) -> DictConfig:
        """Get feature configuration."""
        if self._feature_config is None:
            self._feature_config = load_feature_config()
        return self._feature_config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (dot notation supported)."""
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default

    def get_model_params(self, model_type: str) -> dict[str, Any]:
        """Get parameters for a specific model type."""
        params = OmegaConf.to_container(self.model_params.get(model_type, {}))
        return dict(params) if params else {}

    @property
    def mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI."""
        return os.getenv(
            "MLFLOW_TRACKING_URI",
            self.get("mlflow.tracking_uri", ""),
        )

    @property
    def experiment_name(self) -> str:
        """Get MLflow experiment name."""
        return self.get("mlflow.experiment_name", "hull-tactical-experiments")

    @property
    def target_column(self) -> str:
        """Get target column name."""
        return self.get("data.target_column", "market_forward_excess_returns")

    @property
    def random_state(self) -> int:
        """Get random state for reproducibility."""
        return self.get("training.random_state", 42)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self.config)
