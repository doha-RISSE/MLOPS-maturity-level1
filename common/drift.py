"""Distribution-drift computations for raw Telco features."""
from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from common.schema import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compare raw features using KS for numbers and chi-square/TVD for categories."""
    settings = cfg["validation"]
    feature_results: dict[str, dict[str, Any]] = {}
    for feature in FEATURE_COLUMNS:
        if feature in NUMERIC_FEATURES:
            ref = pd.to_numeric(reference[feature], errors="coerce").dropna()
            cur = pd.to_numeric(current[feature], errors="coerce").dropna()
            if ref.empty or cur.empty:
                statistic, p_value = 0.0, 1.0
            else:
                test = ks_2samp(ref, cur)
                statistic, p_value = float(test.statistic), float(test.pvalue)
            drifted = p_value < settings["drift_alpha"] and statistic >= settings["numeric_min_ks_statistic"]
            feature_results[feature] = {"type": "numeric", "test": "ks_2samp", "statistic": statistic, "p_value": p_value, "drifted": bool(drifted)}
        else:
            ref = reference[feature].astype("string").fillna("<missing>")
            cur = current[feature].astype("string").fillna("<missing>")
            categories = ref.value_counts().index.union(cur.value_counts().index)
            counts_ref = ref.value_counts().reindex(categories, fill_value=0)
            counts_cur = cur.value_counts().reindex(categories, fill_value=0)
            keep = (counts_ref + counts_cur) > 0
            table = pd.DataFrame([counts_ref[keep].to_numpy(), counts_cur[keep].to_numpy()])
            if table.shape[1] < 2 or table.to_numpy().sum() == 0:
                p_value = 1.0
            else:
                _, p_value, _, _ = chi2_contingency(table)
                p_value = float(p_value)
            ref_freq = counts_ref / max(int(counts_ref.sum()), 1)
            cur_freq = counts_cur / max(int(counts_cur.sum()), 1)
            statistic = float(0.5 * (ref_freq - cur_freq).abs().sum())
            drifted = p_value < settings["drift_alpha"] and statistic >= settings["categorical_min_tvd"]
            feature_results[feature] = {"type": "categorical", "test": "chi2_tvd", "statistic": statistic, "p_value": p_value, "drifted": bool(drifted)}
    n_drifted = sum(result["drifted"] for result in feature_results.values())
    return {"features": feature_results, "n_drifted": n_drifted, "share": n_drifted / len(FEATURE_COLUMNS), "detected": n_drifted >= settings["min_drifted_features"]}
