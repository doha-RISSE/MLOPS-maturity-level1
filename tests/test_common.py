"""Unit tests for schema, preprocessing, drift, and promotion rules."""
from __future__ import annotations

import copy

from common.drift import compute_drift
from common.preprocessing import clean_raw, split_data
from common.schema import validate_schema
from pipeline.run_pipeline import load_config
from pipeline.steps.model_validation import decide_promotion
from tests.conftest import make_telco_like_df


def test_schema_reports_all_issue_types() -> None:
    """Schema validation combines independent anomalies in one result."""
    frame = make_telco_like_df(30)
    frame["promo_code"] = "X"
    frame = frame.drop(columns=["Contract"])
    frame.loc[0, "PaymentMethod"] = "Bitcoin"
    types = {issue.type for issue in validate_schema(frame).issues}
    assert types == {"unexpected_features", "missing_features", "unexpected_values"}


def test_total_charges_blanks_are_allowed() -> None:
    """The known blank TotalCharges source convention remains valid."""
    frame = make_telco_like_df(20)
    assert not validate_schema(frame).has_issues


def test_clean_and_split_are_deterministic() -> None:
    """Cleaning maps targets and splitting remains repeatable and disjoint."""
    cleaned = clean_raw(make_telco_like_df(200))
    cfg = load_config("config.yaml")
    one, two = split_data(cleaned, cfg), split_data(cleaned, cfg)
    assert set(one[0].index).isdisjoint(one[1].index)
    assert one[0].index.equals(two[0].index)
    assert set(cleaned["Churn"]) == {0, 1}


def test_drift_and_promotion() -> None:
    """A strong numeric displacement is detected and strict promotion holds."""
    cfg = load_config("config.yaml")
    reference = clean_raw(make_telco_like_df(200)).drop(columns=["Churn"])
    current = reference.copy(); current["MonthlyCharges"] += 500
    assert compute_drift(reference, current, cfg)["features"]["MonthlyCharges"]["drifted"]
    assert decide_promotion(0.8, None, 0) == (True, "first model")
    assert not decide_promotion(0.8, 0.8, 0)[0]
