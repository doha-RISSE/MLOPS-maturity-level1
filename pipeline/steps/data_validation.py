"""Raw schema and distribution validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from common.drift import compute_drift
from common.preprocessing import clean_raw
from common.schema import FEATURE_COLUMNS, validate_schema

__version__ = "1.0.0"


def validate_batch(df: pd.DataFrame, cfg: dict[str, Any], reference: pd.DataFrame | None = None, force: bool = False) -> dict[str, Any]:
    """Produce schema, drift, decision, and reason for an incoming batch."""
    schema = validate_schema(df)
    result: dict[str, Any] = {"schema": schema.to_dict(), "drift": None, "decision": "STOP", "reason": "schema skew"}
    if schema.has_issues:
        return result
    if force:
        result.update({"decision": "TRAIN", "reason": "forced"})
        return result
    if reference is None:
        result.update({"decision": "TRAIN", "reason": "bootstrap : no champion yet"})
        return result
    current = clean_raw(df)[FEATURE_COLUMNS]
    drift = compute_drift(reference[FEATURE_COLUMNS], current, cfg)
    result["drift"] = drift
    if drift["detected"]:
        features = [name for name, item in drift["features"].items() if item["drifted"]]
        result.update({"decision": "TRAIN", "reason": f"data values skew : {', '.join(features)}"})
    else:
        result.update({"decision": "NO_RETRAIN", "reason": "no data values skew"})
    return result


def save_report(report: dict[str, Any], path: Path) -> None:
    """Write a JSON data-validation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
