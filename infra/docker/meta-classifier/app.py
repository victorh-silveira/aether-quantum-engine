from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


logger = logging.getLogger("META")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
FEATURE_DIM = 39


class _HealthcheckAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


def _configure_service_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(_HealthcheckAccessFilter())
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


_configure_service_logging()


class MetaPredictPayload(BaseModel):
    tcn_probability: float = Field(ge=0.0, le=1.0)
    direction: str
    feature_vector: list[float]
    symbol: str = ""


class MetaPredictResult(BaseModel):
    predicted_payoff_edge: float
    meta_applied: bool


app = FastAPI(title="Aether Meta-Regressor", version="2.0.0")
_model_bundle: dict[str, Any] | None = None


def _normalize_features(feature_vector: list[float]) -> np.ndarray:
    values = [float(v) for v in feature_vector]
    if len(values) < FEATURE_DIM:
        values.extend([0.0] * (FEATURE_DIM - len(values)))
    if len(values) > FEATURE_DIM:
        values = values[:FEATURE_DIM]
    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def _load_model_bundle() -> dict[str, Any] | None:
    if not MODELS_DIR.is_dir():
        return None
    candidates = sorted(MODELS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            logger.warning("Falha ao carregar modelo %s: %s", path, exc)
            continue
        if isinstance(bundle, dict) and bundle.get("model") is not None:
            logger.info("Modelo meta-regressor carregado: %s", path.name)
            return bundle
    return None


@app.on_event("startup")
async def startup_load_model() -> None:
    global _model_bundle
    _model_bundle = _load_model_bundle()


@app.get("/health")
async def health() -> dict[str, bool | str]:
    return {"ready": True, "model_loaded": _model_bundle is not None}


@app.post("/v2/predict_meta", response_model=MetaPredictResult)
async def predict_meta(payload: MetaPredictPayload) -> MetaPredictResult:
    bundle = _model_bundle
    if bundle is None:
        return MetaPredictResult(predicted_payoff_edge=0.0, meta_applied=False)
    model = bundle["model"]
    features = _normalize_features(payload.feature_vector)
    try:
        raw = model.predict(features)
        edge = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
    except Exception as exc:
        logger.warning("Inferencia meta-regressor falhou: %s", exc)
        return MetaPredictResult(predicted_payoff_edge=0.0, meta_applied=False)
    return MetaPredictResult(predicted_payoff_edge=edge, meta_applied=True)
