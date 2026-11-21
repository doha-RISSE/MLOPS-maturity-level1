"""CLI entry point for the complete Telco batch training pipeline."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient

from pipeline.steps.alerts import create_schema_alert
from pipeline.steps.data_preparation import prepare_data
from pipeline.steps.data_validation import save_report, validate_batch
from pipeline.steps.evaluate import evaluate_model
from pipeline.steps.metadata import ExecutionMetadata, ensure_experiment
from pipeline.steps.model_validation import save_comparison, validate_model
from pipeline.steps.registry import log_training_provenance, register_training_model
from pipeline.steps.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)

COMPONENT_VERSIONS = {
    "data_validation": "1.0.0", "data_preparation": "1.0.0", "training": "1.0.0",
    "evaluation": "1.0.0", "model_validation": "1.0.0", "registry": "1.0.0", "metadata": "1.0.0", "alerts": "1.0.0",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _champion(client: MlflowClient, cfg: dict[str, Any]) -> Any | None:
    """Look up the champion; its absence is valid during bootstrap."""
    try:
        return client.get_model_version_by_alias(cfg["mlflow"]["registered_model_name"], cfg["mlflow"]["champion_alias"])
    except Exception as error:
        LOGGER.info("No champion available: %s", error)
        return None


def _download_reference(run_id: str) -> pd.DataFrame:
    """Download the raw feature reference of an existing champion."""
    downloaded = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="reference/reference_features.parquet")
    return pd.read_parquet(downloaded)


def _reload_api() -> None:
    """Request a model reload without making a completed run fail on outage."""
    import requests
    url = os.environ.get("API_RELOAD_URL")
    if not url:
        return
    try:
        response = requests.post(url, headers={"X-Admin-Token": os.environ.get("ADMIN_TOKEN", "")}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        LOGGER.warning("Champion reload request failed: %s", error)


def run(input_path: str | Path, cfg: dict[str, Any], force: bool = False) -> str:
    """Execute one batch training run and return its final documented status."""
    input_text = str(input_path)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", mlflow.get_tracking_uri()))
    client = MlflowClient()
    execution_experiment = ensure_experiment(client, cfg["mlflow"]["execution_experiment"])
    metadata = ExecutionMetadata(client, execution_experiment, input_text, force, cfg)
    metadata.set_tags({f"component.{name}.version": version for name, version in COMPONENT_VERSIONS.items()})
    previous = _champion(client, cfg)
    base_tags = {"model.previous_champion_version": previous.version if previous else "none", "model.previous_champion_run_id": previous.run_id if previous else "none"}
    workspace = Path(tempfile.mkdtemp(prefix="telco-pipeline-"))
    try:
        raw = pd.read_csv(input_text, dtype=str, keep_default_na=False)
        client.set_tag(metadata.run_id, "input.rows", str(len(raw)))
        reference = _download_reference(previous.run_id) if previous is not None else None
        with metadata.timed("data_validation"):
            report = validate_batch(raw, cfg, reference, force)
            report_path = workspace / "data_validation" / "report.json"
            save_report(report, report_path)
            metadata.artifact(report_path, "data_validation")
            client.set_tag(metadata.run_id, "artifact.data_validation_report", "data_validation/report.json")
        client.set_tag(metadata.run_id, "decision.reason", report["reason"])
        if report["decision"] == "STOP":
            alert_path = create_schema_alert(metadata.run_id, input_text, report["schema"]["issues"], os.environ.get("ALERTS_DIR", "/app/alerts"))
            artifact_alert = workspace / "alerts" / "alert.json"
            artifact_alert.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(alert_path, artifact_alert)
            metadata.artifact(artifact_alert, "alerts")
            client.set_tag(metadata.run_id, "artifact.alert", "alerts/alert.json")
            metadata.finish("STOPPED_SCHEMA_SKEW", base_tags)
            return "STOPPED_SCHEMA_SKEW"
        if report["decision"] == "NO_RETRAIN":
            metadata.finish("NO_RETRAIN", base_tags)
            return "NO_RETRAIN"
        with metadata.timed("data_preparation"):
            X_train, X_val, X_test, y_train, y_val, y_test, clean = prepare_data(raw, cfg)
        training_experiment = ensure_experiment(client, cfg["mlflow"]["training_experiment"])
        with mlflow.start_run(experiment_id=training_experiment) as training_run:
            with metadata.timed("training"):
                model = train_model(X_train, y_train, cfg)
                log_training_provenance(input_text, raw, clean, (X_train, X_val, X_test, y_train, y_val, y_test), cfg, workspace / "provenance")
            with metadata.timed("evaluation"):
                metrics = {"train": evaluate_model(model, X_train, y_train), "val": evaluate_model(model, X_val, y_val), "test": evaluate_model(model, X_test, y_test)}
                for partition, values in metrics.items():
                    mlflow.log_metrics({f"{partition}_{metric}": value for metric, value in values.items()})
                    if partition in {"train", "test"}:
                        for metric, value in values.items():
                            client.log_metric(metadata.run_id, f"{partition}_{metric}", value)
            with metadata.timed("model_validation"):
                champion_model = mlflow.sklearn.load_model(f"models:/{cfg['mlflow']['registered_model_name']}@{cfg['mlflow']['champion_alias']}") if previous else None
                comparison = validate_model(model, champion_model, X_test, y_test, cfg["model_validation"]["primary_metric"], float(cfg["model_validation"]["min_improvement"]), previous.version if previous else None)
                primary = comparison["primary_metric"]
                if comparison["champion_metrics"]:
                    value = comparison["champion_metrics"][primary]
                    mlflow.log_metric(f"champion_test_{primary}", value)
                    mlflow.log_metric(f"delta_{primary}", comparison["candidate_metrics"][primary] - value)
                comparison_path = workspace / "model_validation" / "comparison.json"
                save_comparison(comparison, comparison_path)
                metadata.artifact(comparison_path, "model_validation")
                client.set_tag(metadata.run_id, "artifact.model_validation", "model_validation/comparison.json")
            with metadata.timed("registry"):
                registered = register_training_model(model, X_train, cfg["mlflow"]["registered_model_name"], client)
                validation_status = "PASSED" if comparison["promote"] else "FAILED"
                for key, value in {"validation_status": validation_status, "primary_metric": comparison["primary_metric"], "compared_to_version": comparison["champion_version"], "pipeline_execution_run_id": metadata.run_id}.items():
                    client.set_model_version_tag(registered.name, registered.version, key, value)
                model_uri = f"models:/{registered.name}/{registered.version}"
                metadata.set_tags({"model.new_version": registered.version, "model.promoted": str(comparison["promote"]).lower(), "training_run_id": training_run.info.run_id, "artifact.model_uri": model_uri})
                if comparison["promote"]:
                    client.set_registered_model_alias(registered.name, cfg["mlflow"]["champion_alias"], registered.version)
                    status = "SUCCESS_PROMOTED"
                else:
                    status = "SUCCESS_NOT_PROMOTED"
        metadata.finish(status, base_tags)
        if status == "SUCCESS_PROMOTED":
            _reload_api()
        return status
    except Exception:
        error_path = workspace / "errors" / "traceback.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        LOGGER.exception("Pipeline failed")
        try:
            metadata.artifact(error_path, "errors")
            metadata.finish("FAILED", {**base_tags, "artifact.error": "errors/traceback.txt"})
        except Exception:
            LOGGER.exception("Could not finalize failed execution metadata")
        return "FAILED"


def main() -> int:
    """Parse CLI options and map final statuses to required exit codes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config_path = args.config or os.environ.get("CONFIG_PATH", "/app/config.yaml" if Path("/app/config.yaml").exists() else "config.yaml")
    status = run(args.input, load_config(config_path), args.force)
    LOGGER.info("Pipeline final status: %s", status)
    return {"STOPPED_SCHEMA_SKEW": 2, "FAILED": 1}.get(status, 0)


if __name__ == "__main__":
    raise SystemExit(main())
