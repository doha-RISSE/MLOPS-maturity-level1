"""Thread-safe request-window drift monitoring."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import pandas as pd

from app.metrics import DRIFT_DETECTED, DRIFT_FEATURE, DRIFT_READY, DRIFT_SCORE, DRIFT_SHARE, DRIFT_WINDOW
from common.drift import compute_drift
from common.schema import FEATURE_COLUMNS


class DriftMonitor:
    """Retain recent valid payloads and update metrics during scrapes."""
    def __init__(self, cfg: dict[str, Any]) -> None:
        serving = cfg["serving"]
        self.cfg, self.minimum, self.refresh = cfg, int(serving["drift_min_samples"]), float(serving["drift_refresh_seconds"])
        self.window: deque[dict[str, Any]] = deque(maxlen=int(serving["drift_window_size"]))
        self.lock, self.last_refresh = threading.Lock(), 0.0
        self.emitted_features: set[str] = set()

    def add(self, payload: dict[str, Any]) -> None:
        """Add an already validated API payload."""
        with self.lock:
            self.window.append({key: payload[key] for key in FEATURE_COLUMNS})

    def refresh_metrics(self, reference: pd.DataFrame | None) -> None:
        """Recalculate only when due and enough data/reference exist."""
        with self.lock:
            now = time.monotonic()
            if now - self.last_refresh < self.refresh:
                return
            self.last_refresh = now
            current = pd.DataFrame(list(self.window), columns=FEATURE_COLUMNS)
        DRIFT_WINDOW.set(len(current))
        if reference is None or len(current) < self.minimum:
            DRIFT_READY.set(0); DRIFT_SHARE.set(0); DRIFT_DETECTED.set(0)
            return
        outcome = compute_drift(reference, current, self.cfg)
        DRIFT_READY.set(1); DRIFT_SHARE.set(outcome["share"]); DRIFT_DETECTED.set(int(outcome["detected"]))
        for feature, result in outcome["features"].items():
            DRIFT_SCORE.labels(feature).set(result["statistic"])
            DRIFT_FEATURE.labels(feature).set(int(result["drifted"]))
            self.emitted_features.add(feature)
