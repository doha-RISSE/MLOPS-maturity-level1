"""Reusable valid synthetic Telco data."""
from __future__ import annotations

import pandas as pd

from common.schema import ALLOWED_VALUES, RAW_COLUMNS


def make_telco_like_df(n: int, seed: int = 42) -> pd.DataFrame:
    """Create schema-valid, class-balanced source-shaped Telco data."""
    rows: list[dict[str, object]] = []
    for index in range(n):
        row: dict[str, object] = {"customerID": f"customer-{seed}-{index}", "tenure": str(index % 73), "MonthlyCharges": str(20 + (index % 80)), "TotalCharges": " " if index % 31 == 0 else str(100 + index * 8), "Churn": "Yes" if index % 4 == 0 else "No"}
        for column, choices in ALLOWED_VALUES.items():
            if column not in row:
                values = sorted(choices, key=str)
                row[column] = str(values[index % len(values)])
        rows.append(row)
    return pd.DataFrame(rows)[RAW_COLUMNS]
