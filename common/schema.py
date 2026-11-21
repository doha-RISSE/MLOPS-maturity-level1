"""The single source of truth for Telco CSV schema rules."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
RAW_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [column for column in FEATURE_COLUMNS if column not in NUMERIC_FEATURES]
ALLOWED_VALUES: dict[str, set[Any]] = {
    "gender": {"Male", "Female"}, "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"}, "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"}, "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
    "Churn": {"Yes", "No"},
}
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "tenure": (0, 120), "MonthlyCharges": (0, 1000), "TotalCharges": (0, 100000),
}


@dataclass(frozen=True)
class SchemaIssue:
    """A schema issue found before model processing."""
    type: str
    columns: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this issue."""
        return asdict(self)


@dataclass(frozen=True)
class SchemaReport:
    """Complete validation result."""
    issues: list[SchemaIssue]

    @property
    def has_issues(self) -> bool:
        """Whether any issue was found."""
        return bool(self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this report."""
        return {"issues": [issue.to_dict() for issue in self.issues]}


def _invalid_details(series: pd.Series, invalid: pd.Series) -> dict[str, Any]:
    values = series[invalid]
    examples = [str(value) for value in values.drop_duplicates().head(5).tolist()]
    return {"count": int(invalid.sum()), "examples": examples}


def _stripped(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def validate_schema(df: pd.DataFrame) -> SchemaReport:
    """Validate all raw columns while reporting every anomaly at once."""
    issues: list[SchemaIssue] = []
    unexpected = [column for column in df.columns if column not in RAW_COLUMNS]
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if unexpected:
        issues.append(SchemaIssue("unexpected_features", unexpected, {"count": len(unexpected)}))
    if missing:
        issues.append(SchemaIssue("missing_features", missing, {"count": len(missing)}))

    invalid_columns: list[str] = []
    invalid_details: dict[str, Any] = {}
    for column in RAW_COLUMNS:
        if column not in df.columns or column == ID_COLUMN:
            continue
        source = df[column]
        if column in ALLOWED_VALUES:
            cleaned = _stripped(source)
            allowed = {str(value) for value in ALLOWED_VALUES[column]}
            invalid = source.isna() | ~cleaned.isin(allowed)
        else:
            cleaned = _stripped(source)
            numeric = pd.to_numeric(cleaned, errors="coerce")
            lower, upper = NUMERIC_RANGES[column]
            if column == "TotalCharges":
                blank = source.isna() | cleaned.isna() | cleaned.eq("")
                invalid = ~blank & (numeric.isna() | ~numeric.between(lower, upper))
            else:
                invalid = source.isna() | numeric.isna() | ~numeric.between(lower, upper)
        if bool(invalid.any()):
            invalid_columns.append(column)
            invalid_details[column] = _invalid_details(source, invalid)
    if invalid_columns:
        issues.append(SchemaIssue("unexpected_values", invalid_columns, invalid_details))
    return SchemaReport(issues)
