"""Deterministic cleaning, feature engineering, and train-only preprocessing."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS, ID_COLUMN, NUMERIC_FEATURES, TARGET


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data without fitting any learned statistic."""
    result = df.copy().drop_duplicates(subset=[ID_COLUMN], keep="first").drop(columns=[ID_COLUMN])
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype("string").str.strip()
    result["tenure"] = pd.to_numeric(result["tenure"], errors="raise")
    result["MonthlyCharges"] = pd.to_numeric(result["MonthlyCharges"], errors="raise")
    result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
    result["SeniorCitizen"] = pd.to_numeric(result["SeniorCitizen"], errors="raise").astype(int)
    result[TARGET] = result[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return result[[*FEATURE_COLUMNS, TARGET]]


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add stateless features identically in training and serving."""
    result = X.loc[:, FEATURE_COLUMNS].copy()
    result["tenure_bin"] = pd.cut(
        result["tenure"], bins=[0, 12, 24, 48, float("inf")], right=False,
        labels=["0-11", "12-23", "24-47", "48+"],
    ).astype("string")
    return result


def build_preprocessor() -> ColumnTransformer:
    """Build the unfitted preprocessing transformer."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, [*CATEGORICAL_FEATURES, "tenure_bin"])],
        remainder="drop",
    )


def split_data(df_clean: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create deterministic stratified train, validation, and test datasets."""
    random_state = int(cfg["data"]["random_state"])
    test_size = float(cfg["data"]["split"]["test"])
    validation_size = float(cfg["data"]["split"]["validation"])
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[TARGET]
    X_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    validation_relative = validation_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_remaining, y_remaining, test_size=validation_relative, stratify=y_remaining, random_state=random_state)
    return engineer_features(X_train), engineer_features(X_val), engineer_features(X_test), y_train, y_val, y_test
