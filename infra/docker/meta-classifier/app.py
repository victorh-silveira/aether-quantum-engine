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


class MetaPredictPayload(BaseModel):
    tcn_probability: float = Field(ge=0.0, le=1.0)
    direction: str
    feature_vector: list[float]
    symbol: str = ""


class MetaPredictResult(BaseModel):
    calibrated_payoff_score: float = Field(ge=0.0, le=1.0)
    meta_applied: bool


app = FastAPI(title="Aether Meta-Classificador", version="1.0.0")
_model_bundle: dict[str, Any] | None = None


def _side_score(tcn_probability: float, direction: str) -> float:
    prob = float(tcn_probability)
    side = str(direction or "").upper()
    if side == "PUT":
        return max(0.0, min(1.0, 1.0 - prob))
    return max(0.0, min(1.0, prob))


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
            logger.info("Modelo meta-classificador carregado: %s", path.name)
            return bundle
    return None


def _blend_scores(tcn_side: float, meta_prob: float, weight: float) -> float:
    w = max(0.0, min(1.0, float(weight)))
    blended = (1.0 - w) * float(tcn_side) + w * float(meta_prob)
    return max(0.0, min(1.0, blended))


@app.on_event("startup")
async def startup_load_model() -> None:
    global _model_bundle
    _model_bundle = _load_model_bundle()


@app.get("/health")
async def health() -> dict[str, bool | str]:
    return {"ready": True, "model_loaded": _model_bundle is not None}


@app.post("/v2/predict_meta", response_model=MetaPredictResult)
async def predict_meta(payload: MetaPredictPayload) -> MetaPredictResult:
    tcn_side = _side_score(payload.tcn_probability, payload.direction)
    bundle = _model_bundle
    if bundle is None:
        return MetaPredictResult(calibrated_payoff_score=tcn_side, meta_applied=False)
    model = bundle["model"]
    weight = float(bundle.get("blend_weight", 0.55))
    features = _normalize_features(payload.feature_vector)
    try:
        raw = model.predict_proba(features)[0]
        meta_prob = float(raw[1]) if len(raw) > 1 else float(raw[0])
    except Exception as exc:
        logger.warning("Inferencia meta-classificador falhou: %s", exc)
        return MetaPredictResult(calibrated_payoff_score=tcn_side, meta_applied=False)
    score = _blend_scores(tcn_side, meta_prob, weight)
    return MetaPredictResult(calibrated_payoff_score=score, meta_applied=True)
