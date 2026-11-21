"""Send reproducible normal, invalid, and optional failure traffic to the API."""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.schema import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)


def traffic(url: str, source: Path, rate: float, duration: float, invalid_fraction: float, fail_fraction: float, seed: int) -> None:
    """Send rate-limited source rows as requests for a bounded duration."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    rng = random.Random(seed)
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    rows = frame.to_dict("records")
    deadline, index = time.monotonic() + duration, 0
    while time.monotonic() < deadline:
        row = rows[index % len(rows)]; index += 1
        payload = {column: (None if column == "TotalCharges" and str(row[column]).strip() == "" else row[column]) for column in FEATURE_COLUMNS}
        if rng.random() < invalid_fraction:
            payload["unexpected"] = "invalid"
        try:
            requests.post(f"{url.rstrip('/')}/predict", json=payload, timeout=10)
            if fail_fraction and rng.random() < fail_fraction:
                requests.post(f"{url.rstrip('/')}/admin/debug/fail", headers={"X-Admin-Token": os.environ.get("ADMIN_TOKEN", "")}, timeout=10)
        except requests.RequestException as error:
            LOGGER.warning("Traffic request failed: %s", error)
        time.sleep(max(0, 1 / rate))


def main() -> None:
    """Run the traffic generator from CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--rate", type=float, default=5)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--invalid-fraction", type=float, default=0.05)
    parser.add_argument("--fail-fraction", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    traffic(args.url, args.source, args.rate, args.duration, args.invalid_fraction, args.fail_fraction, args.seed)


if __name__ == "__main__":
    main()
