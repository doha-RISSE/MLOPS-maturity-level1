"""Model-fitting pipeline step."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from common.preprocessing import build_preprocessor

__version__ = "1.0.0"


def train_model(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict[str, Any]) -> Pipeline:
    """Fit the complete train-only scikit-learn pipeline."""
    model = Pipeline([("preprocessor", build_preprocessor()), ("classifier", RandomForestClassifier(**cfg["model"]["params"]))])
    model.fit(X_train, y_train)
    return model
