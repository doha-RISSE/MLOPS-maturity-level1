"""Prometheus metric definitions for the API."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "handler", "status"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Feature preparation and prediction latency", buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5])
MODEL_LOADED = Gauge("model_loaded", "Whether a champion model is loaded")
MODEL_VERSION = Gauge("model_version_info", "Loaded model version", ["model_name", "version", "run_id"])
DRIFT_SCORE = Gauge("data_drift_score", "Per feature drift score", ["feature"])
DRIFT_FEATURE = Gauge("data_drift_feature_detected", "Per feature drift detection", ["feature"])
DRIFT_SHARE = Gauge("data_drift_share", "Share of drifted features")
DRIFT_DETECTED = Gauge("data_drift_detected", "Whether drift is detected")
DRIFT_WINDOW = Gauge("drift_window_samples", "Number of payloads in drift window")
DRIFT_READY = Gauge("data_drift_ready", "Whether drift has enough reference and samples")
