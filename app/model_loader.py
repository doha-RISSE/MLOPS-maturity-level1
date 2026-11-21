"""Champion-model loading backed by the MLflow registry."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from app.metrics import MODEL_LOADED, MODEL_VERSION

LOGGER = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """An in-memory champion model with registry provenance."""
    model: Any
    name: str
    version: str
    run_id: str
    reference: pd.DataFrame


class ModelLoader:
    """Load and atomically replace a served champion model."""
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.loaded: LoadedModel | None = None
        self._labels: tuple[str, str, str] | None = None

    def reload(self) -> bool:
        """Fetch model and its reference artifact; retain old model on failure."""
        try:
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri()))
            client = MlflowClient()
            name = self.cfg["mlflow"]["registered_model_name"]
            alias = self.cfg["mlflow"]["champion_alias"]
            version = client.get_model_version_by_alias(name, alias)
            model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
            path = mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path="reference/reference_features.parquet")
            reference = pd.read_parquet(path)
            self.loaded = LoadedModel(model, name, version.version, version.run_id, reference)
            if self._labels:
                MODEL_VERSION.remove(*self._labels)
            self._labels = (name, version.version, version.run_id)
            MODEL_LOADED.set(1)
            MODEL_VERSION.labels(*self._labels).set(1)
            return True
        except Exception as error:
            LOGGER.warning("Champion model is unavailable: %s", error)
            MODEL_LOADED.set(1 if self.loaded else 0)
            return False
