"""Data cleaning and splitting pipeline step."""
from __future__ import annotations

from typing import Any

import pandas as pd

from common.preprocessing import clean_raw, split_data

__version__ = "1.0.0"


def prepare_data(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Clean an input batch and return training partitions and clean raw data."""
    clean = clean_raw(df)
    return (*split_data(clean, cfg), clean)
