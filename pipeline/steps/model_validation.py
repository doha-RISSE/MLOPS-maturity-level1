"""Candidate-versus-champion model validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.steps.evaluate import evaluate_model

__version__ = "1.0.0"


def decide_promotion(new_value: float, champion_value_or_none: float | None, min_improvement: float) -> tuple[bool, str]:
    """Strictly decide whether a candidate improves the production model."""
    if champion_value_or_none is None:
        return True, "first model"
    if new_value > champion_value_or_none + min_improvement:
        return True, f"candidate improves champion by {new_value - champion_value_or_none:.6f}"
    return False, "candidate does not strictly exceed champion improvement threshold"


def validate_model(candidate: Any, champion: Any | None, X_test: Any, y_test: Any, primary_metric: str, min_improvement: float, champion_version: str | None) -> dict[str, Any]:
    """Evaluate candidate and optional champion on the same current test partition."""
    candidate_metrics = evaluate_model(candidate, X_test, y_test)
    champion_metrics = evaluate_model(champion, X_test, y_test) if champion is not None else None
    champion_value = None if champion_metrics is None else champion_metrics[primary_metric]
    promote, reason = decide_promotion(candidate_metrics[primary_metric], champion_value, min_improvement)
    return {"primary_metric": primary_metric, "candidate_metrics": candidate_metrics, "champion_metrics": champion_metrics, "champion_version": champion_version or "none", "promote": promote, "reason": reason}


def save_comparison(comparison: dict[str, Any], path: Path) -> None:
    """Write a model validation artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
