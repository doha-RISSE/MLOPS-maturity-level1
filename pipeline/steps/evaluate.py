"""Evaluation pipeline step."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

__version__ = "1.0.0"

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calculate all required classification metrics at a fixed 0.5 threshold."""
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }
