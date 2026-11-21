"""Create deterministic demo batches from the user-provided Telco CSV."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)


def make_batches(input_path: Path, output: Path, seed: int) -> None:
    """Create six disjoint/derived batches required for the demonstration."""
    if not input_path.exists():
        raise SystemExit(f"Source CSV is missing. Place it at {input_path}")
    source = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if len(source) < 7000:
        raise SystemExit("Source CSV must contain at least 7000 rows")
    shuffled = source.sample(frac=1, random_state=seed).reset_index(drop=True)
    initial, normal, drift_source = shuffled.iloc[:4000].copy(), shuffled.iloc[4000:5500].copy(), shuffled.iloc[5500:7000].copy()
    monthly = pd.to_numeric(drift_source["MonthlyCharges"], errors="coerce") * 1.35
    tenure = (pd.to_numeric(drift_source["tenure"], errors="coerce") * 0.5).round()
    drift_source["MonthlyCharges"] = monthly.round(2).map(lambda value: f"{value:.2f}")
    drift_source["tenure"] = tenure.astype("Int64").astype(str)
    drift_source["TotalCharges"] = (tenure * monthly).round(2).map(lambda value: f"{value:.2f}")
    broken = normal.iloc[:500].copy()
    extra = broken.copy(); extra["promo_code"] = "PROMO"
    missing = broken.drop(columns=["Contract"])
    bad = broken.copy(); bad.loc[bad.index[:100], "PaymentMethod"] = "Bitcoin"; bad.loc[bad.index[:50], "tenure"] = "-5"
    output.mkdir(parents=True, exist_ok=True)
    for name, batch in {"batch_initial.csv": initial, "batch_normal.csv": normal, "batch_drift.csv": drift_source, "batch_broken_extra_column.csv": extra, "batch_broken_missing_column.csv": missing, "batch_broken_bad_values.csv": bad}.items():
        batch.to_csv(output / name, index=False)
    LOGGER.info("Created six batches in %s", output)


def main() -> None:
    """Run batch creation from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_batches(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
