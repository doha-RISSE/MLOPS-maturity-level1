"""Execution-run metadata helpers, independent from model training runs."""
from __future__ import annotations

import hashlib
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import mlflow
from mlflow import MlflowClient

__version__ = "1.0.0"


def sha256_file(path: str | Path) -> str:
    """Return a file's stable SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(prefix: str, value: Any) -> dict[str, str]:
    """Flatten nested configuration to MLflow-compatible string parameters."""
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for key, child in value.items():
            result.update(flatten(f"{prefix}.{key}" if prefix else str(key), child))
        return result
    return {prefix: str(value).lower() if isinstance(value, bool) else str(value)}


class ExecutionMetadata:
    """Small facade around an MLflow execution run created through MlflowClient."""
    def __init__(self, client: MlflowClient, experiment_id: str, input_path: str, force: bool, cfg: dict[str, Any]) -> None:
        self.client, self.run = client, client.create_run(experiment_id)
        self.started = datetime.now(timezone.utc)
        self.durations: dict[str, float] = {}
        self.set_tags({
            "pipeline.version": str(cfg["pipeline_version"]), "pipeline.git_commit": __import__("os").environ.get("GIT_COMMIT", "unknown"),
            "executor.user": __import__("getpass").getuser(), "executor.host": socket.gethostname(),
            "executor.environment": __import__("os").environ.get("EXECUTOR", "docker-compose"), "started_at_utc": self.started.isoformat(),
            "input.path": input_path, "input.sha256": sha256_file(input_path),
        })
        for key, value in {"cli.input": input_path, "cli.force": str(force).lower(), **flatten("config", cfg)}.items():
            self.client.log_param(self.run.info.run_id, key, value)

    @property
    def run_id(self) -> str:
        """Expose the execution run identifier."""
        return self.run.info.run_id

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        """Time one pipeline step."""
        started = perf_counter()
        try:
            yield
        finally:
            value = perf_counter() - started
            self.durations[name] = value
            self.client.log_metric(self.run_id, f"duration_seconds.{name}", value)

    def artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Upload one local artifact."""
        self.client.log_artifact(self.run_id, str(path), artifact_path)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set several tags through the documented single-tag client API."""
        for key, value in tags.items():
            self.client.set_tag(self.run_id, key, value)

    def finish(self, status: str, extra_tags: dict[str, str] | None = None) -> None:
        """End the execution run with its aggregate duration and tags."""
        ended = datetime.now(timezone.utc)
        tags = {"status": status, "ended_at_utc": ended.isoformat(), **(extra_tags or {})}
        self.set_tags(tags)
        self.client.log_metric(self.run_id, "duration_seconds.total", (ended - self.started).total_seconds())
        self.client.set_terminated(self.run_id, status="FAILED" if status == "FAILED" else "FINISHED")


def ensure_experiment(client: MlflowClient, name: str) -> str:
    """Find or create an experiment and return its ID."""
    existing = client.get_experiment_by_name(name)
    return existing.experiment_id if existing else client.create_experiment(name)
