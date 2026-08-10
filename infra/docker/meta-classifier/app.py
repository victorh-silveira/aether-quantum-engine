from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from learn_runtime import (
    fit_regressor,
    load_learn_buffer,
    persist_regressor_bundle,
    save_learn_buffer,
    should_retrain_meta,
)


logger = logging.getLogger("META")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
META_FEATURE_DIM = 43
DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    *(f"feature_{index}" for index in range(34)),
    "micro_bid_ask_spread_momentum",
    "micro_bid_ask_spread_momentum_zscore",
    "volatility_shadow_ratio",
    "volatility_shadow_ratio_zscore",
    "cross_symbol_prob_delta",
    "cross_symbol_vol_ratio_diff",
    "cross_symbol_rsi_spread",
    "micro_tick_acceleration",
    "keltner_deviation_ratio",
)


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


class PredictMetaRequest(BaseModel):
    tcn_probability: float = Field(ge=0.0, le=1.0)
    direction: str
    feature_vector: list[float]
    symbol: str = ""

    @model_validator(mode="before")
    @classmethod
    def map_features_key(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "feature_vector" not in normalized and "features" in normalized:
            normalized["feature_vector"] = normalized["features"]
        return normalized


class MetaPredictResult(BaseModel):
    predicted_payoff_edge: float
    meta_applied: bool
    edge_expectancy: str
    model_version: str = ""


class LearnMetaRequest(BaseModel):
    feature_vector: list[float]
    target: float
    contract_id: str = ""
    symbol: str = ""


def _classify_edge_expectancy(edge: float) -> str:
    if edge <= 0.0:
        return "LOSS_EXPECTED"
    if edge < 0.04:
        return "NO_EDGE_NEUTRAL"
    return "WIN_EXPECTED"


app = FastAPI(title="Aether Meta-Regressor", version="2.1.0")
_model_bundle: dict[str, Any] | None = None
_model_path: Path | None = None
_model_mtime: float = 0.0
_model_load_error: str | None = None
_n_loaded: int = 0
_buffer_x: list[list[float]] = []
_buffer_y: list[float] = []
_lock = threading.Lock()
RETRAIN_MIN_N = int(os.getenv("META_RETRAIN_MIN_N", "1"))
MAX_BUFFER = int(os.getenv("META_MAX_BUFFER", "2000"))
BUFFER_PATH = MODELS_DIR / "meta_learn_buffer.pkl"


def _resolve_feature_names(bundle: dict[str, Any]) -> list[str]:
    stored = bundle.get("feature_names")
    if isinstance(stored, list) and stored:
        names = [str(name) for name in stored]
        if names != list(DEFAULT_FEATURE_NAMES):
            logger.warning(
                "feature_names do bundle divergem do schema canônico 43D; usando nomes do artefato",
            )
        return names
    return list(DEFAULT_FEATURE_NAMES)


def _build_feature_dataframe(bundle: dict[str, Any], feature_vector: list[float]) -> pd.DataFrame:
    names = _resolve_feature_names(bundle)
    expected = len(names)
    if len(feature_vector) != expected:
        raise ValueError(f"feature_vector deve ter {expected} elementos, recebeu {len(feature_vector)}")
    row = [float(value) for value in feature_vector]
    frame = pd.DataFrame([row], columns=names)
    return frame.loc[:, names]


def _model_version() -> str:
    if _model_path is not None:
        return _model_path.name
    return str((_model_bundle or {}).get("model_version") or "none")


def _load_model_bundle() -> dict[str, Any] | None:
    global _model_load_error, _model_path, _model_mtime, _n_loaded
    failures: list[str] = []
    if not MODELS_DIR.is_dir():
        _model_load_error = f"diretorio de modelos ausente: {MODELS_DIR}"
        return None
    candidates = sorted(MODELS_DIR.glob("*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        _model_load_error = f"nenhum artefato .pkl encontrado em {MODELS_DIR}"
        return None
    for path in candidates:
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            message = f"{path.name}: {exc}"
            failures.append(message)
            logger.warning("Falha ao carregar modelo %s: %s", path, exc)
            continue
        if not isinstance(bundle, dict) or bundle.get("model") is None:
            failures.append(f"{path.name}: bundle sem chave model")
            continue
        model = bundle["model"]
        model_type = str(bundle.get("model_type") or "regressor")
        if model_type != "regressor":
            failures.append(f"{path.name}: model_type={model_type}")
            logger.warning("Artefato %s ignorado: model_type=%s", path.name, model_type)
            continue
        if not callable(getattr(model, "predict", None)):
            failures.append(f"{path.name}: metodo predict ausente")
            logger.warning("Artefato %s sem metodo predict", path.name)
            continue
        _model_load_error = None
        _model_path = path
        _model_mtime = float(path.stat().st_mtime)
        _n_loaded += 1
        feature_count = len(_resolve_feature_names(bundle))
        logger.info("Modelo meta-regressor carregado: %s | feature_dim=%d", path.name, feature_count)
        return bundle
    _model_load_error = "; ".join(failures) if failures else f"nenhum regressor valido em {MODELS_DIR}"
    return None


def _maybe_hot_reload() -> None:
    global _model_bundle, _model_mtime
    if not MODELS_DIR.is_dir():
        return
    candidates = sorted(MODELS_DIR.glob("*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return
    latest = candidates[0]
    mtime = float(latest.stat().st_mtime)
    if _model_bundle is not None and mtime <= float(_model_mtime) + 1e-9:
        return
    bundle = _load_model_bundle()
    if bundle is not None:
        _model_bundle = bundle
        logger.info("Hot-reload meta: %s", latest.name)


def _regressor_unavailable_detail() -> str:
    if _model_load_error:
        return f"LGBMRegressor indisponivel: {_model_load_error}"
    return "LGBMRegressor indisponivel: modelo nao carregado no bootstrap"


@app.on_event("startup")
async def startup_load_model() -> None:
    global _model_bundle, _buffer_x, _buffer_y
    _model_bundle = _load_model_bundle()
    if _model_bundle is None:
        logger.error(_regressor_unavailable_detail())
    xs, ys = load_learn_buffer(BUFFER_PATH)
    _buffer_x = xs
    _buffer_y = ys


@app.get("/health")
async def health() -> dict[str, Any]:
    _maybe_hot_reload()
    names = _resolve_feature_names(_model_bundle) if _model_bundle is not None else list(DEFAULT_FEATURE_NAMES)
    return {
        "ready": _model_bundle is not None,
        "model_loaded": _model_bundle is not None,
        "feature_dim": len(names),
        "model_path": str(_model_path) if _model_path else "",
        "model_mtime": float(_model_mtime),
        "model_version": _model_version(),
        "n_loaded": int(_n_loaded),
        "buffer_n": len(_buffer_y),
        "load_error": _model_load_error or "",
    }


@app.get("/version")
async def version() -> dict[str, Any]:
    _maybe_hot_reload()
    return {
        "model_version": _model_version(),
        "model_path": str(_model_path) if _model_path else "",
        "feature_dim": META_FEATURE_DIM,
        "ready": _model_bundle is not None,
    }


@app.post("/v2/predict_meta", response_model=MetaPredictResult)
async def predict_meta(payload: PredictMetaRequest) -> MetaPredictResult:
    _maybe_hot_reload()
    bundle = _model_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail=_regressor_unavailable_detail())
    model = bundle["model"]
    try:
        input_features_dataframe = _build_feature_dataframe(bundle, payload.feature_vector)
        raw_edge = model.predict(input_features_dataframe)[0]
        edge = float(raw_edge)
    except Exception as exc:
        logger.warning("Inferencia meta-regressor falhou: %s", exc)
        return MetaPredictResult(
            predicted_payoff_edge=0.0,
            meta_applied=False,
            edge_expectancy="LOSS_EXPECTED",
            model_version=_model_version(),
        )
    return MetaPredictResult(
        predicted_payoff_edge=edge,
        meta_applied=True,
        edge_expectancy=_classify_edge_expectancy(edge),
        model_version=_model_version(),
    )


@app.post("/v1/learn")
async def learn(payload: LearnMetaRequest) -> dict[str, Any]:
    global _model_bundle, _model_path, _model_mtime
    vector = [float(v) for v in payload.feature_vector]
    if len(vector) != META_FEATURE_DIM:
        raise HTTPException(status_code=400, detail=f"feature_vector deve ter {META_FEATURE_DIM}")
    retrained = False
    detail = "buffered"
    with _lock:
        _buffer_x.append(vector)
        _buffer_y.append(float(payload.target))
        if len(_buffer_y) > int(MAX_BUFFER):
            overflow = len(_buffer_y) - int(MAX_BUFFER)
            del _buffer_x[:overflow]
            del _buffer_y[:overflow]
        save_learn_buffer(BUFFER_PATH, _buffer_x, _buffer_y)
        n = len(_buffer_y)
        do_fit = should_retrain_meta(buffer_n=n, retrain_min_n=int(RETRAIN_MIN_N))
        xs = list(_buffer_x)
        ys = list(_buffer_y)
        names = _resolve_feature_names(_model_bundle) if _model_bundle is not None else list(DEFAULT_FEATURE_NAMES)
    if do_fit:
        try:
            model = fit_regressor(xs, ys)
            path = persist_regressor_bundle(
                MODELS_DIR,
                model,
                n_train=len(ys),
                feature_names=names,
                feature_dim=META_FEATURE_DIM,
            )
            with _lock:
                _model_bundle = {
                    "model": model,
                    "model_type": "regressor",
                    "feature_names": names,
                    "n_train": len(ys),
                    "auto_learn_applied": True,
                    "model_version": path.name,
                    "feature_dim": META_FEATURE_DIM,
                }
                _model_path = path
                _model_mtime = float(path.stat().st_mtime)
            retrained = True
            detail = "ok"
            logger.info("META learn fit n=%d path=%s", len(ys), path.name)
        except Exception as exc:
            detail = str(exc)
            logger.warning("META learn fit falhou: %s", exc)
    return {
        "ok": True,
        "buffer_n": n,
        "retrained": retrained,
        "n_train": n,
        "retrain_detail": detail,
        "model_version": _model_version(),
    }
