"""Hyperparameter optimization for Hull Tactical."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseModel, ModelFactory
from src.training.validator import WalkForwardValidator
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.

    Supports optimization for all model types with
    walk-forward validation as the objective.
    """

    def __init__(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = 3600,
        direction: str = "minimize",
        metric: str = "rmse",
    ):
        """
        Initialize optimizer.

        Args:
            n_trials: Number of optimization trials.
            timeout: Timeout in seconds.
            direction: 'minimize' or 'maximize'.
            metric: Metric to optimize.
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.metric = metric
        self.study = None
        self.best_params: Dict[str, Any] = {}

    def _get_search_space(self, model_type: str) -> Dict[str, Any]:
        """Get search space for a model type."""
        spaces = {
            "lightgbm": {
                "n_estimators": ("int", 100, 1000),
                "learning_rate": ("float", 0.01, 0.3, "log"),
                "max_depth": ("int", 3, 10),
                "num_leaves": ("int", 15, 127),
                "min_child_samples": ("int", 5, 50),
                "subsample": ("float", 0.5, 1.0),
                "colsample_bytree": ("float", 0.5, 1.0),
                "reg_alpha": ("float", 1e-8, 10.0, "log"),
                "reg_lambda": ("float", 1e-8, 10.0, "log"),
            },
            "xgboost": {
                "n_estimators": ("int", 100, 1000),
                "learning_rate": ("float", 0.01, 0.3, "log"),
                "max_depth": ("int", 3, 10),
                "min_child_weight": ("int", 1, 10),
                "subsample": ("float", 0.5, 1.0),
                "colsample_bytree": ("float", 0.5, 1.0),
                "reg_alpha": ("float", 1e-8, 10.0, "log"),
                "reg_lambda": ("float", 1e-8, 10.0, "log"),
            },
            "catboost": {
                "iterations": ("int", 100, 1000),
                "learning_rate": ("float", 0.01, 0.3, "log"),
                "depth": ("int", 3, 10),
                "l2_leaf_reg": ("float", 1e-8, 10.0, "log"),
            },
            "random_forest": {
                "n_estimators": ("int", 100, 500),
                "max_depth": ("int", 5, 20),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 10),
            },
        }

        return spaces.get(model_type, {})

    def _sample_params(self, trial, model_type: str) -> Dict[str, Any]:
        """Sample hyperparameters for a trial."""
        space = self._get_search_space(model_type)
        params = {}

        for param_name, param_spec in space.items():
            if param_spec[0] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_spec[1], param_spec[2]
                )
            elif param_spec[0] == "float":
                if len(param_spec) > 3 and param_spec[3] == "log":
                    params[param_name] = trial.suggest_float(
                        param_name, param_spec[1], param_spec[2], log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, param_spec[1], param_spec[2]
                    )
            elif param_spec[0] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_spec[1]
                )

        # Add fixed params
        config = Config()
        fixed_params = config.get_model_params(model_type)
        for key in ["random_state", "verbosity", "verbose", "n_jobs"]:
            if key in fixed_params:
                params[key] = fixed_params[key]

        return params

    def optimize(
        self,
        model_type: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_splits: int = 3,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model.

        Args:
            model_type: Type of model to optimize.
            df: Training DataFrame.
            feature_cols: Feature column names.
            target_col: Target column name.
            n_splits: Number of CV splits.

        Returns:
            Dictionary with best parameters and score.
        """
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")

        validator = WalkForwardValidator()

        def objective(trial) -> float:
            """Objective function for optimization."""
            params = self._sample_params(trial, model_type)

            def model_factory():
                return ModelFactory.create(model_type, params=params)

            try:
                results = validator.validate(
                    df=df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    model_factory=model_factory,
                    n_splits=n_splits,
                )

                score = results["mean_metrics"].get(self.metric, float("inf"))

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("inf") if self.direction == "minimize" else float("-inf")

        # Create study
        self.study = optuna.create_study(direction=self.direction)

        # Run optimization
        logger.info(f"Starting hyperparameter optimization for {model_type}")
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        # Get best params
        self.best_params = self._sample_params(self.study.best_trial, model_type)

        logger.info(f"Best {self.metric}: {self.study.best_value}")
        logger.info(f"Best params: {self.best_params}")

        return {
            "best_params": self.best_params,
            "best_score": self.study.best_value,
            "n_trials": len(self.study.trials),
        }

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        trials = []
        for trial in self.study.trials:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                **trial.params,
            }
            trials.append(trial_data)

        return pd.DataFrame(trials)

    def get_param_importance(self) -> Dict[str, float]:
        """Get parameter importance scores."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        try:
            import optuna

            return optuna.importance.get_param_importances(self.study)
        except Exception as e:
            logger.warning(f"Could not compute param importance: {e}")
            return {}
