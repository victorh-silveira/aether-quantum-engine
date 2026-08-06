from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from buffer_io import buffer_class_counts, load_learn_buffer, save_learn_buffer
from learn_policy import retrain_min_for_label, should_retrain_after_learn
from runtime import fit_classifier, load_latest_classifier, persist_bundle, predict_p_loss


logger = logging.getLogger("LOSS_CLF")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
FEATURE_DIM = int(os.getenv("LOSS_FEATURE_DIM", "24"))
READY_N = int(os.getenv("LOSS_READY_N", "24"))
RETRAIN_MIN_N = int(os.getenv("LOSS_RETRAIN_MIN_N", "24"))
RETRAIN_ON_LOSS_MIN_N = int(os.getenv("LOSS_RETRAIN_ON_LOSS_MIN_N", "2"))
MAX_BUFFER = int(os.getenv("LOSS_MAX_BUFFER", "2000"))
VETO_P_LOSS_FLOOR = float(os.getenv("LOSS_VETO_P_LOSS_FLOOR", "0.65"))
FEATURE_NAMES = tuple(f"f_{index}" for index in range(FEATURE_DIM))


class _HealthcheckAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


def _configure_service_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("uvicorn.access").addFilter(_HealthcheckAccessFilter())
    logging.getLogger("uvicorn").setLevel(logging.WARNING)


_configure_service_logging()


class PredictLossRequest(BaseModel):
    feature_vector: list[float]
    symbol: str = ""
    direction: str = ""
    veto_p_loss_floor: float | None = None


class LossPredictResult(BaseModel):
    p_loss: float
    veto: bool
    auto_learn_applied: bool
    model_version: str
    n_train: int
    veto_ready: bool


class LearnRequest(BaseModel):
    feature_vector: list[float]
    label: str = Field(description="WIN ou LOSS")
    contract_id: str = ""
    symbol: str = ""


class RetrainResult(BaseModel):
    ok: bool
    n_train: int
    model_version: str
    detail: str = ""


app = FastAPI(title="Aether Loss-Classifier", version="1.0.0")
_lock = threading.RLock()
_model: Any | None = None
_model_path: Path | None = None
_model_mtime: float = 0.0
_model_version: str = "none"
_n_train: int = 0
_auto_learn_applied: bool = False
_buffer_x: list[list[float]] = []
_buffer_y: list[int] = []
_load_error: str = ""


def _validate_vector(vector: list[float]) -> list[float]:
    if len(vector) != FEATURE_DIM:
        raise ValueError(f"feature_vector deve ter {FEATURE_DIM} elementos, recebeu {len(vector)}")
    return [float(value) for value in vector]


def _veto_ready() -> bool:
    return _model is not None and int(_n_train) >= int(READY_N)


def _persist_buffer_unlocked() -> None:
    save_learn_buffer(MODELS_DIR, _buffer_x, _buffer_y)


def _load_buffer_unlocked() -> None:
    global _buffer_x, _buffer_y
    loaded = load_learn_buffer(MODELS_DIR)
    if loaded is None:
        return
    _buffer_x, _buffer_y = loaded
    if len(_buffer_y) > int(MAX_BUFFER):
        overflow = len(_buffer_y) - int(MAX_BUFFER)
        del _buffer_x[:overflow]
        del _buffer_y[:overflow]
    logger.info("Buffer learn carregado n=%d %s", len(_buffer_y), buffer_class_counts(_buffer_y))


def _apply_bundle(bundle: dict[str, Any], path: Path | None, *, auto_learn: bool) -> None:
    global _model, _model_path, _model_mtime, _model_version, _n_train, _auto_learn_applied, _load_error
    _model = bundle["model"]
    _model_path = path
    _model_mtime = float(path.stat().st_mtime) if path is not None and path.is_file() else 0.0
    _n_train = int(bundle.get("n_train") or 0)
    _model_version = str(bundle.get("model_version") or (path.name if path else "memory"))
    _auto_learn_applied = bool(auto_learn or bundle.get("auto_learn_applied"))
    _load_error = ""


def _load_latest_model() -> bool:
    global _load_error
    loaded = load_latest_classifier(MODELS_DIR)
    if loaded is None:
        _load_error = f"nenhum classifier em {MODELS_DIR}"
        return False
    bundle, path = loaded
    _apply_bundle(bundle, path, auto_learn=bool(bundle.get("auto_learn_applied")))
    logger.info("Modelo loss carregado: %s n_train=%d", path.name, _n_train)
    return True


def _maybe_hot_reload() -> None:
    loaded = load_latest_classifier(MODELS_DIR)
    if loaded is None:
        return
    bundle, path = loaded
    mtime = float(path.stat().st_mtime)
    if _model is not None and mtime <= float(_model_mtime) + 1e-9:
        return
    _apply_bundle(bundle, path, auto_learn=bool(bundle.get("auto_learn_applied")))
    logger.info("Hot-reload loss: %s", path.name)


def _fit_from_buffer(*, min_n: int | None = None) -> RetrainResult:
    floor = int(min_n) if min_n is not None else int(RETRAIN_MIN_N)
    with _lock:
        if len(_buffer_y) < floor:
            return RetrainResult(ok=False, n_train=len(_buffer_y), model_version=_model_version, detail=f"n<{floor}")
        if len(set(_buffer_y)) < 2:
            return RetrainResult(
                ok=False, n_train=len(_buffer_y), model_version=_model_version, detail="precisa WIN e LOSS"
            )
        model = fit_classifier(_buffer_x, _buffer_y)
        path = persist_bundle(
            MODELS_DIR, model, len(_buffer_y), FEATURE_NAMES, FEATURE_DIM, auto_learn=True
        )
        _apply_bundle(
            {"model": model, "n_train": len(_buffer_y), "model_version": path.stem, "auto_learn_applied": True},
            path,
            auto_learn=True,
        )
        _persist_buffer_unlocked()
        logger.info("Retrain loss ok n=%d ver=%s", _n_train, _model_version)
        return RetrainResult(ok=True, n_train=_n_train, model_version=_model_version, detail="ok")


@app.on_event("startup")
async def startup() -> None:
    with _lock:
        _load_buffer_unlocked()
        _load_latest_model()


@app.get("/health")
async def health() -> dict[str, Any]:
    with _lock:
        _maybe_hot_reload()
        counts = buffer_class_counts(_buffer_y)
        return {
            "ready": True,
            "model_loaded": _model is not None,
            "veto_ready": _veto_ready(),
            "feature_dim": FEATURE_DIM,
            "n_train": int(_n_train),
            "model_version": _model_version,
            "model_path": str(_model_path) if _model_path else "",
            "model_mtime": float(_model_mtime),
            "auto_learn_applied": bool(_auto_learn_applied),
            "buffer_n": len(_buffer_y),
            "buffer_win": int(counts["win"]),
            "buffer_loss": int(counts["loss"]),
            "load_error": _load_error,
        }


@app.get("/version")
async def version() -> dict[str, Any]:
    with _lock:
        return {
            "model_version": _model_version,
            "n_train": int(_n_train),
            "auto_learn_applied": bool(_auto_learn_applied),
            "veto_ready": _veto_ready(),
        }


@app.post("/v1/predict_loss", response_model=LossPredictResult)
async def predict_loss(payload: PredictLossRequest) -> LossPredictResult:
    with _lock:
        _maybe_hot_reload()
        try:
            vector = _validate_vector(payload.feature_vector)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        floor = float(payload.veto_p_loss_floor) if payload.veto_p_loss_floor is not None else VETO_P_LOSS_FLOOR
        if _model is None:
            return LossPredictResult(
                p_loss=0.5,
                veto=False,
                auto_learn_applied=False,
                model_version=_model_version,
                n_train=int(_n_train),
                veto_ready=False,
            )
        try:
            p_loss = predict_p_loss(_model, vector)
        except Exception as exc:
            logger.warning("predict falhou: %s", exc)
            return LossPredictResult(
                p_loss=0.5,
                veto=False,
                auto_learn_applied=bool(_auto_learn_applied),
                model_version=_model_version,
                n_train=int(_n_train),
                veto_ready=_veto_ready(),
            )
        ready = _veto_ready()
        return LossPredictResult(
            p_loss=p_loss,
            veto=bool(ready and p_loss + 1e-12 >= float(floor)),
            auto_learn_applied=bool(_auto_learn_applied),
            model_version=_model_version,
            n_train=int(_n_train),
            veto_ready=ready,
        )


@app.post("/v1/learn")
async def learn(payload: LearnRequest) -> dict[str, Any]:
    label = str(payload.label or "").strip().upper()
    if label not in {"WIN", "LOSS"}:
        raise HTTPException(status_code=400, detail="label deve ser WIN ou LOSS")
    try:
        vector = _validate_vector(payload.feature_vector)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    retrain_result: RetrainResult | None = None
    fit_min = retrain_min_for_label(
        label=label,
        retrain_min_n=int(RETRAIN_MIN_N),
        retrain_on_loss_min_n=int(RETRAIN_ON_LOSS_MIN_N),
    )
    with _lock:
        _buffer_x.append(vector)
        _buffer_y.append(1 if label == "LOSS" else 0)
        if len(_buffer_y) > int(MAX_BUFFER):
            overflow = len(_buffer_y) - int(MAX_BUFFER)
            del _buffer_x[:overflow]
            del _buffer_y[:overflow]
        should_retrain = should_retrain_after_learn(
            label=label,
            buffer_n=len(_buffer_y),
            retrain_min_n=int(RETRAIN_MIN_N),
            retrain_on_loss_min_n=int(RETRAIN_ON_LOSS_MIN_N),
        )
        _persist_buffer_unlocked()
    if should_retrain:
        retrain_result = _fit_from_buffer(min_n=fit_min)
    with _lock:
        return {
            "ok": True,
            "buffer_n": len(_buffer_y),
            "retrained": bool(retrain_result and retrain_result.ok),
            "model_version": _model_version,
            "n_train": int(_n_train),
            "auto_learn_applied": bool(_auto_learn_applied),
            "retrain_detail": str(retrain_result.detail) if retrain_result is not None else "",
        }


@app.post("/v1/retrain", response_model=RetrainResult)
async def retrain() -> RetrainResult:
    return _fit_from_buffer()
