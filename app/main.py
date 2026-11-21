"""FastAPI inference service for the registered Telco churn champion."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.drift_monitor import DriftMonitor
from app.metrics import HTTP_REQUESTS, PREDICTION_LATENCY
from app.model_loader import ModelLoader
from app.schemas import CustomerFeatures
from common.preprocessing import engineer_features
from common.schema import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load app configuration from the mounted default path."""
    selected = path or os.environ.get("CONFIG_PATH", "/app/config.yaml" if Path("/app/config.yaml").exists() else "config.yaml")
    with Path(selected).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _handler(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def _authorized(request: Request) -> bool:
    return request.headers.get("X-Admin-Token") == os.environ.get("ADMIN_TOKEN", "")


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    """Construct the API application, allowing test configuration injection."""
    cfg = config or load_config()
    loader = ModelLoader(cfg)
    monitor = DriftMonitor(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader.reload()
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.loader, app.state.monitor = loader, monitor

    @app.middleware("http")
    async def instrument(request: Request, call_next: Any) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception("Unhandled API exception")
            response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        HTTP_REQUESTS.labels(request.method, _handler(request), str(response.status_code)).inc()
        return response

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Return liveness independently from registry availability."""
        return {"status": "ok", "model_loaded": loader.loaded is not None}

    @app.post("/predict")
    async def predict(features: CustomerFeatures) -> dict[str, Any]:
        """Score one valid customer while keeping blocking sklearn work off-loop."""
        if loader.loaded is None:
            raise HTTPException(status_code=503, detail="No champion model is loaded")
        raw = features.model_dump()
        monitor.add(raw)
        started = perf_counter()
        frame = pd.DataFrame([[raw[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        engineered = engineer_features(frame)
        probability = float((await asyncio.to_thread(loader.loaded.model.predict_proba, engineered))[0, 1])
        PREDICTION_LATENCY.observe(perf_counter() - started)
        return {"churn_probability": probability, "churn_prediction": int(probability >= 0.5), "model_name": loader.loaded.name, "model_version": loader.loaded.version}

    @app.get("/metrics")
    async def metrics() -> Response:
        """Refresh in-memory drift state and expose Prometheus metrics."""
        monitor.refresh_metrics(loader.loaded.reference if loader.loaded else None)
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/reload")
    async def reload_model(request: Request) -> dict[str, bool]:
        """Reload the champion model after a promotion."""
        if not _authorized(request):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {"reloaded": loader.reload()}

    @app.post("/admin/debug/fail")
    async def debug_fail(request: Request) -> None:
        """Optionally exercise 500 monitoring during demonstrations only."""
        if not _authorized(request):
            raise HTTPException(status_code=401, detail="Unauthorized")
        if os.environ.get("ENABLE_DEBUG_ENDPOINTS", "false").lower() != "true":
            raise HTTPException(status_code=404, detail="Not found")
        raise RuntimeError("Debug failure requested")

    return app


app = create_app()
