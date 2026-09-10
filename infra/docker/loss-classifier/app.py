from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_ML_ROOT = Path(__file__).resolve().parent.parent
if (_ML_ROOT / "ml_common" / "__init__.py").is_file() and str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from ml_common.async_fit import run_in_thread
from ml_common.learn_ids import ContractIdDedupe
from ml_common.names import LOSS_FEATURE_NAMES
from ml_common.schema import canonical_schema_hash, request_schema_hash_ok, validate_feature_vector

from buffer_io import buffer_class_counts, load_learn_buffer, save_learn_buffer
from learn_policy import (
    bootstrap_seed_keep_detail,
    retrain_min_for_label,
    retrain_skipped_reason,
    should_retrain_after_learn,
)
from calib import binary_ece, buffer_p_loss, fit_temperature
from runtime import (
    fit_classifier,
    is_bootstrap_bundle,
    is_collapsed_classifier,
    is_degenerate_quality,
    is_stale_gaussian_bootstrap,
    load_latest_classifier,
    persist_bundle,
    predict_p_loss,
    seed_bootstrap_classifier,
)


logger = logging.getLogger("LOSS_CLF")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
FEATURE_DIM = int(os.getenv("LOSS_FEATURE_DIM", "24"))
READY_N = int(os.getenv("LOSS_READY_N", "32"))
RETRAIN_MIN_N = int(os.getenv("LOSS_RETRAIN_MIN_N", "12"))
RETRAIN_ON_LOSS_MIN_N = int(os.getenv("LOSS_RETRAIN_ON_LOSS_MIN_N", "4"))
BOOTSTRAP_EXIT_N = int(os.getenv("LOSS_BOOTSTRAP_EXIT_N", "12"))
MAX_BUFFER = int(os.getenv("LOSS_MAX_BUFFER", "2000"))
MIN_WIN_FOR_LOSS_RETRAIN = int(os.getenv("LOSS_MIN_WIN_FOR_LOSS_RETRAIN", "4"))
VETO_P_LOSS_FLOOR = float(os.getenv("LOSS_VETO_P_LOSS_FLOOR", "0.58"))
YOUNG_TEMP_N = int(os.getenv("LOSS_YOUNG_TEMP_N", "32"))
ECE_MAX = float(os.getenv("LOSS_ECE_DEGENERATE", "0.35"))
FEATURE_NAMES = LOSS_FEATURE_NAMES if int(FEATURE_DIM) == len(LOSS_FEATURE_NAMES) else tuple(
    f"f_{index}" for index in range(FEATURE_DIM)
)
SCHEMA_HASH = canonical_schema_hash(FEATURE_NAMES)


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
    schema_hash: str = ""


class LossPredictResult(BaseModel):
    p_loss: float
    veto: bool
    auto_learn_applied: bool
    model_version: str
    n_train: int
    veto_ready: bool
    bootstrap: bool = False
    collapsed: bool = False


class LearnRequest(BaseModel):
    feature_vector: list[float]
    label: str = Field(description="WIN ou LOSS")
    contract_id: str = ""
    symbol: str = ""
    schema_hash: str = ""


class RetrainResult(BaseModel):
    ok: bool
    n_train: int
    model_version: str
    detail: str = ""


app = FastAPI(title="Aether Loss-Classifier", version="1.0.0")
_lock = threading.RLock()
_learn_ids = ContractIdDedupe()
_model: Any | None = None
_model_path: Path | None = None
_model_mtime: float = 0.0
_model_version: str = "none"
_n_train: int = 0
_auto_learn_applied: bool = False
_bootstrap: bool = False
_degenerate: bool = False
_collapsed: bool = False
_buffer_x: list[list[float]] = []
_buffer_y: list[int] = []
_load_error: str = ""
_cal_temperature: float = 1.0
_cal_ece: float = 1.0


def _validate_vector(vector: list[float]) -> list[float]:
    return validate_feature_vector(vector, FEATURE_DIM)


def _require_schema_hash(payload_hash: str) -> None:
    if not request_schema_hash_ok(payload_hash, SCHEMA_HASH):
        raise HTTPException(status_code=400, detail="schema_hash divergente")


def _model_source() -> str:
    return "online" if bool(_auto_learn_applied) else "offline"


def _refresh_quality_unlocked() -> None:
    global _degenerate, _collapsed
    collapsed_now = bool(_collapsed)
    if _model is not None and len(_buffer_x) >= 8:
        collapsed_now = is_collapsed_classifier(_model, _buffer_x)
        _collapsed = collapsed_now
    _degenerate = is_degenerate_quality(
        collapsed=collapsed_now,
        cal_ece=float(_cal_ece),
        n_train=int(_n_train),
        ece_max=float(ECE_MAX),
        mature_n=int(YOUNG_TEMP_N),
        bootstrap=bool(_bootstrap),
    )


def _veto_ready() -> bool:
    if _model is None or bool(_degenerate):
        return False
    return int(_n_train) >= int(READY_N)


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
    global _model, _model_path, _model_mtime, _model_version, _n_train
    global _auto_learn_applied, _bootstrap, _degenerate, _collapsed, _load_error
    global _cal_temperature, _cal_ece
    _model = bundle["model"]
    _model_path = path
    _model_mtime = float(path.stat().st_mtime) if path is not None and path.is_file() else 0.0
    _n_train = int(bundle.get("n_train") or 0)
    _model_version = str(bundle.get("model_version") or (path.name if path else "memory"))
    _auto_learn_applied = bool(auto_learn or bundle.get("auto_learn_applied"))
    _bootstrap = False if _auto_learn_applied else is_bootstrap_bundle(bundle, version=_model_version)
    _degenerate = bool(bundle.get("degenerate"))
    _collapsed = bool(bundle.get("collapsed", False))
    _cal_temperature = float(bundle.get("cal_temperature") or 1.0)
    _cal_ece = float(bundle.get("cal_ece") or 1.0)
    _load_error = ""


def _load_latest_model() -> bool:
    global _load_error
    loaded = load_latest_classifier(MODELS_DIR, expected_schema_hash=SCHEMA_HASH)
    if loaded is None:
        _load_error = f"nenhum classifier em {MODELS_DIR}"
        return False
    bundle, path = loaded
    _apply_bundle(bundle, path, auto_learn=bool(bundle.get("auto_learn_applied")))
    _refresh_quality_unlocked()
    logger.info("Modelo loss carregado: %s n_train=%d", path.name, _n_train)
    return True


def _maybe_hot_reload() -> None:
    loaded = load_latest_classifier(MODELS_DIR, expected_schema_hash=SCHEMA_HASH)
    if loaded is None:
        return
    bundle, path = loaded
    mtime = float(path.stat().st_mtime)
    if _model is not None and mtime <= float(_model_mtime) + 1e-9:
        return
    _apply_bundle(bundle, path, auto_learn=bool(bundle.get("auto_learn_applied")))
    _refresh_quality_unlocked()
    logger.info("Hot-reload loss: %s", path.name)


def _fit_from_buffer(*, min_n: int | None = None) -> RetrainResult:
    floor = int(min_n) if min_n is not None else int(RETRAIN_MIN_N)
    with _lock:
        global _collapsed
        keep = bootstrap_seed_keep_detail(
            bootstrap=bool(_bootstrap),
            buffer_n=len(_buffer_y),
            n_classes=len(set(_buffer_y)),
            exit_n=int(BOOTSTRAP_EXIT_N),
            floor=floor,
        )
        if keep is not None:
            return RetrainResult(
                ok=False,
                n_train=len(_buffer_y),
                model_version=_model_version,
                detail=keep,
            )
        if len(_buffer_y) < floor:
            return RetrainResult(ok=False, n_train=len(_buffer_y), model_version=_model_version, detail=f"n<{floor}")
        if len(set(_buffer_y)) < 2:
            return RetrainResult(
                ok=False, n_train=len(_buffer_y), model_version=_model_version, detail="precisa WIN e LOSS"
            )
        model = fit_classifier(_buffer_x, _buffer_y)
        if is_collapsed_classifier(model, _buffer_x):
            _collapsed = True
            logger.warning("Retrain loss rejeitado colapso n=%d", len(_buffer_y))
            return RetrainResult(
                ok=False,
                n_train=len(_buffer_y),
                model_version=_model_version,
                detail="collapsed_reject",
            )
        cal_t = fit_temperature(model, _buffer_x, _buffer_y)
        cal_ece = binary_ece(buffer_p_loss(model, _buffer_x, temperature=cal_t), _buffer_y)
        degenerate = is_degenerate_quality(
            collapsed=False,
            cal_ece=cal_ece,
            n_train=len(_buffer_y),
            ece_max=float(ECE_MAX),
            mature_n=int(YOUNG_TEMP_N),
            bootstrap=False,
        )
        path = persist_bundle(
            MODELS_DIR,
            model,
            len(_buffer_y),
            FEATURE_NAMES,
            FEATURE_DIM,
            auto_learn=True,
            cal_temperature=cal_t,
            cal_ece=cal_ece,
            schema_hash=SCHEMA_HASH,
            degenerate=degenerate,
            collapsed=False,
        )
        _apply_bundle(
            {
                "model": model,
                "n_train": len(_buffer_y),
                "model_version": path.stem,
                "auto_learn_applied": True,
                "bootstrap": False,
                "degenerate": degenerate,
                "collapsed": False,
                "cal_temperature": cal_t,
                "cal_ece": cal_ece,
            },
            path,
            auto_learn=True,
        )
        _collapsed = False
        _persist_buffer_unlocked()
        _refresh_quality_unlocked()
        logger.info(
            "LEARN schema=%s degenerate=%s source=%s n=%d ver=%s",
            SCHEMA_HASH[:12],
            int(_degenerate),
            _model_source(),
            _n_train,
            _model_version,
        )
        return RetrainResult(ok=True, n_train=_n_train, model_version=_model_version, detail="ok")


@app.on_event("startup")
async def startup() -> None:
    with _lock:
        _load_buffer_unlocked()
        loaded = _load_latest_model()
        stale_seed = bool(loaded) and bool(_bootstrap) and is_stale_gaussian_bootstrap(_model_version)
        if loaded and not stale_seed:
            _refresh_quality_unlocked()
            logger.info(
                "HEALTH schema=%s degenerate=%s source=%s n=%d veto_ready=%s",
                SCHEMA_HASH[:12],
                int(_degenerate),
                _model_source(),
                _n_train,
                _veto_ready(),
            )
            return
        if stale_seed:
            logger.info("Reseed loss: bootstrap OOD legado version=%s", _model_version)
        seeded = seed_bootstrap_classifier(MODELS_DIR, FEATURE_DIM, FEATURE_NAMES, schema_hash=SCHEMA_HASH)
        if seeded is None:
            logger.error("Falha ao semear loss-classifier em %s", MODELS_DIR)
            return
        bundle, path = seeded
        _apply_bundle(bundle, path, auto_learn=False)
        _refresh_quality_unlocked()
        logger.info(
            "HEALTH schema=%s degenerate=%s source=%s n=%d veto_ready=%s",
            SCHEMA_HASH[:12],
            int(_degenerate),
            _model_source(),
            _n_train,
            _veto_ready(),
        )


@app.get("/health")
async def health() -> dict[str, Any]:
    with _lock:
        _maybe_hot_reload()
        _refresh_quality_unlocked()
        counts = buffer_class_counts(_buffer_y)
        return {
            "ready": True,
            "model_loaded": _model is not None,
            "veto_ready": _veto_ready(),
            "feature_dim": FEATURE_DIM,
            "schema_hash": SCHEMA_HASH,
            "source": _model_source(),
            "n_train": int(_n_train),
            "model_version": _model_version,
            "model_path": str(_model_path) if _model_path else "",
            "model_mtime": float(_model_mtime),
            "auto_learn_applied": bool(_auto_learn_applied),
            "bootstrap": bool(_bootstrap),
            "degenerate": bool(_degenerate),
            "collapsed": bool(_collapsed),
            "cal_temperature": float(_cal_temperature),
            "cal_ece": float(_cal_ece),
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
    _require_schema_hash(payload.schema_hash)
    with _lock:
        _maybe_hot_reload()
        try:
            vector = _validate_vector(payload.feature_vector)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        floor = float(payload.veto_p_loss_floor) if payload.veto_p_loss_floor is not None else VETO_P_LOSS_FLOOR
        if _model is None:
            raise HTTPException(status_code=503, detail="loss-classifier sem modelo carregado")
        try:
            young = (
                bool(_bootstrap)
                or str(_model_version).startswith("loss_bootstrap_live")
                or int(_n_train) < int(YOUNG_TEMP_N)
            )
            temp = 2.0 if young else float(_cal_temperature)
            p_loss = predict_p_loss(_model, vector, temperature=temp)
        except Exception as exc:
            logger.warning("predict falhou: %s", exc)
            return LossPredictResult(
                p_loss=0.5,
                veto=False,
                auto_learn_applied=bool(_auto_learn_applied),
                model_version=_model_version,
                n_train=int(_n_train),
                veto_ready=_veto_ready(),
                bootstrap=bool(_bootstrap),
                collapsed=bool(_collapsed),
            )
        ready = _veto_ready()
        return LossPredictResult(
            p_loss=p_loss,
            veto=bool(ready and p_loss + 1e-12 >= float(floor)),
            auto_learn_applied=bool(_auto_learn_applied),
            model_version=_model_version,
            n_train=int(_n_train),
            veto_ready=ready,
            bootstrap=bool(_bootstrap),
            collapsed=bool(_collapsed),
        )


@app.post("/v1/learn")
async def learn(payload: LearnRequest) -> dict[str, Any]:
    _require_schema_hash(payload.schema_hash)
    label = str(payload.label or "").strip().upper()
    if label not in {"WIN", "LOSS"}:
        raise HTTPException(status_code=400, detail="label deve ser WIN ou LOSS")
    try:
        vector = _validate_vector(payload.feature_vector)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    retrain_result: RetrainResult | None = None
    skip_reason = "ok"
    duplicate = False
    with _lock:
        if _learn_ids.seen_or_add(payload.contract_id):
            duplicate = True
            skip_reason = "duplicate_contract"
        else:
            _buffer_x.append(vector)
            _buffer_y.append(1 if label == "LOSS" else 0)
            if len(_buffer_y) > int(MAX_BUFFER):
                overflow = len(_buffer_y) - int(MAX_BUFFER)
                del _buffer_x[:overflow]
                del _buffer_y[:overflow]
        counts = buffer_class_counts(_buffer_y)
        boot = bool(_bootstrap)
        fit_min = retrain_min_for_label(
            label=label,
            retrain_min_n=int(RETRAIN_MIN_N),
            retrain_on_loss_min_n=int(RETRAIN_ON_LOSS_MIN_N),
            bootstrap_active=boot,
            bootstrap_exit_n=int(BOOTSTRAP_EXIT_N),
        )
        should_retrain = (not duplicate) and should_retrain_after_learn(
            label=label,
            buffer_n=len(_buffer_y),
            retrain_min_n=int(RETRAIN_MIN_N),
            retrain_on_loss_min_n=int(RETRAIN_ON_LOSS_MIN_N),
            buffer_win=int(counts["win"]),
            buffer_loss=int(counts["loss"]),
            min_win_for_loss_retrain=int(MIN_WIN_FOR_LOSS_RETRAIN),
            bootstrap_active=boot,
            bootstrap_exit_n=int(BOOTSTRAP_EXIT_N),
        )
        skip_reason = (
            skip_reason
            if duplicate
            else retrain_skipped_reason(
                label=label,
                buffer_n=len(_buffer_y),
                retrain_min_n=int(RETRAIN_MIN_N),
                retrain_on_loss_min_n=int(RETRAIN_ON_LOSS_MIN_N),
                buffer_win=int(counts["win"]),
                buffer_loss=int(counts["loss"]),
                min_win_for_loss_retrain=int(MIN_WIN_FOR_LOSS_RETRAIN),
                bootstrap_active=boot,
                bootstrap_exit_n=int(BOOTSTRAP_EXIT_N),
                should_retrain=should_retrain,
            )
        )
        if not duplicate:
            _persist_buffer_unlocked()
    if should_retrain:
        retrain_result = await run_in_thread(_fit_from_buffer, min_n=fit_min)
        if retrain_result is not None and not retrain_result.ok:
            skip_reason = str(retrain_result.detail or "fit_failed")
        elif retrain_result is not None and retrain_result.ok:
            skip_reason = "ok"
    with _lock:
        return {
            "ok": True,
            "buffer_n": len(_buffer_y),
            "retrained": bool(retrain_result and retrain_result.ok),
            "model_version": _model_version,
            "n_train": int(_n_train),
            "auto_learn_applied": bool(_auto_learn_applied),
            "retrain_detail": str(retrain_result.detail) if retrain_result is not None else "",
            "retrain_skipped_reason": skip_reason,
            "schema_hash": SCHEMA_HASH,
            "degenerate": bool(_degenerate),
            "source": _model_source(),
            "duplicate": duplicate,
        }


@app.post("/v1/retrain", response_model=RetrainResult)
async def retrain() -> RetrainResult:
    return await run_in_thread(_fit_from_buffer)
