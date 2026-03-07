"""Model registry interface for MLflow integration."""

from typing import Any

from src.models.base import BaseModel
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Model registry interface for MLflow/Dagshub.

    Provides methods to:
    - Register models to MLflow registry
    - Load models from registry
    - Manage model versions and stages
    """

    def __init__(self, tracking_uri: str | None = None):
        """
        Initialize model registry.

        Args:
            tracking_uri: MLflow tracking URI.
        """
        config = Config()
        self.tracking_uri = tracking_uri or config.mlflow_tracking_uri

        self._mlflow = None
        self._client = None

    @property
    def mlflow(self):
        """Lazy load MLflow."""
        if self._mlflow is None:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow = mlflow
        return self._mlflow

    @property
    def client(self):
        """Get MLflow client."""
        if self._client is None:
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient(self.tracking_uri)
        return self._client

    def register_model(
        self,
        model: BaseModel,
        model_name: str,
        run_id: str | None = None,
        artifact_path: str = "model",
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Register a model to MLflow registry.

        Args:
            model: Model to register.
            model_name: Name for the registered model.
            run_id: MLflow run ID (uses active run if None).
            artifact_path: Path within artifacts.
            tags: Optional tags for the model.

        Returns:
            Model version string.
        """
        if not model.is_fitted:
            raise ValueError("Cannot register unfitted model")

        # Log model to MLflow
        if run_id is None:
            # Use active run or create new one
            active_run = self.mlflow.active_run()
            if active_run is None:
                with self.mlflow.start_run() as run:
                    run_id = run.info.run_id
                    self._log_model_artifacts(model, artifact_path)
            else:
                run_id = active_run.info.run_id
        else:
            self._log_model_artifacts(model, artifact_path)

        # Register the model
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = self.mlflow.register_model(model_uri, model_name)

        # Add tags
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(model_name, result.version, key, value)

        logger.info(f"Registered model {model_name} version {result.version}")
        return result.version

    def _log_model_artifacts(self, model: BaseModel, artifact_path: str) -> None:
        """Log model artifacts to MLflow."""
        import tempfile

        # Log the sklearn/custom model
        self.mlflow.sklearn.log_model(
            model.model,
            artifact_path,
            registered_model_name=None,  # Will register separately
        )

        # Log additional model metadata
        self.mlflow.log_params(model.params)
        self.mlflow.log_metrics(model.training_metrics)

        # Log feature names
        if model.feature_names:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("\n".join(model.feature_names))
                self.mlflow.log_artifact(f.name, artifact_path)

    def load_model(
        self,
        model_name: str,
        version: str | None = None,
        stage: str | None = None,
    ) -> Any:
        """
        Load a model from the registry.

        Args:
            model_name: Name of the registered model.
            version: Specific version to load.
            stage: Stage to load from ('Production', 'Staging', etc.).

        Returns:
            Loaded model.
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"

        logger.info(f"Loading model from {model_uri}")
        return self.mlflow.sklearn.load_model(model_uri)

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """
        Transition a model version to a new stage.

        Args:
            model_name: Name of the registered model.
            version: Version to transition.
            stage: Target stage ('Staging', 'Production', 'Archived').
            archive_existing: Whether to archive existing models in target stage.
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")

    def get_latest_version(
        self,
        model_name: str,
        stages: list[str] | None = None,
    ) -> str | None:
        """
        Get the latest version of a model.

        Args:
            model_name: Name of the registered model.
            stages: Optional list of stages to filter by.

        Returns:
            Latest version string or None.
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=stages)
            if versions:
                return versions[0].version
            return None
        except Exception as e:
            logger.warning(f"Could not get latest version for {model_name}: {e}")
            return None

    def list_models(self) -> list[dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of model information dictionaries.
        """
        models = []
        for rm in self.client.search_registered_models():
            models.append(
                {
                    "name": rm.name,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "status": v.status,
                        }
                        for v in rm.latest_versions
                    ],
                }
            )
        return models

    def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a specific model version."""
        self.client.delete_model_version(model_name, version)
        logger.info(f"Deleted {model_name} v{version}")

    def get_model_version_info(self, model_name: str, version: str) -> dict[str, Any]:
        """
        Get information about a model version.

        Args:
            model_name: Name of the registered model.
            version: Version to get info for.

        Returns:
            Dictionary with model version information.
        """
        mv = self.client.get_model_version(model_name, version)
        return {
            "name": mv.name,
            "version": mv.version,
            "stage": mv.current_stage,
            "status": mv.status,
            "run_id": mv.run_id,
            "source": mv.source,
            "tags": mv.tags,
        }
