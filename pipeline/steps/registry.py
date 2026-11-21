"""MLflow model registration and provenance logging."""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow import MlflowClient

from common.schema import FEATURE_COLUMNS
from pipeline.steps.metadata import flatten, sha256_file

__version__ = "1.0.0"


def log_training_provenance(input_path: str, raw: pd.DataFrame, clean: pd.DataFrame, splits: tuple[Any, ...], cfg: dict[str, Any], workspace: Path) -> None:
    """Log software, data, configuration, and reference artifacts to training run."""
    import sklearn
    workspace.mkdir(parents=True, exist_ok=True)
    requirements = Path("pipeline/requirements.txt")
    dockerfile = Path("pipeline/Dockerfile")
    freeze = workspace / "pip_freeze.txt"
    freeze.write_text(subprocess.check_output(["python", "-m", "pip", "freeze"], text=True), encoding="utf-8")
    mlflow.set_tags({"git_commit": __import__("os").environ.get("GIT_COMMIT", "unknown"), "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "mlflow_version": mlflow.__version__, "data_sha256": sha256_file(input_path), "data_source_path": input_path, "data_n_rows": str(len(raw))})
    mlflow.log_params(flatten("", {"model": cfg["model"]["params"], "data": cfg["data"], "validation": cfg["validation"], "model_validation": cfg["model_validation"]}))
    mlflow.log_artifact(str(requirements), "environment")
    mlflow.log_artifact(str(dockerfile), "environment")
    mlflow.log_artifact(str(freeze), "environment")
    raw.to_csv(workspace / "raw_batch.csv", index=False)
    mlflow.log_artifact(str(workspace / "raw_batch.csv"), "data")
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    for name, X, y in (("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)):
        X.assign(Churn=y).to_parquet(workspace / f"{name}.parquet", index=False)
        mlflow.log_artifact(str(workspace / f"{name}.parquet"), "data")
    clean[FEATURE_COLUMNS].loc[X_train.index].to_parquet(workspace / "reference_features.parquet", index=False)
    mlflow.log_artifact(str(workspace / "reference_features.parquet"), "reference")


def register_training_model(model: Any, X_train: pd.DataFrame, model_name: str, client: MlflowClient) -> Any:
    """Log a complete sklearn pipeline, then register its new model version."""
    mlflow.sklearn.log_model(model, artifact_path="model", signature=infer_signature(X_train, model.predict(X_train)), input_example=X_train.head(5))
    version = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)
    return client.get_model_version(model_name, version.version)
