from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

import joblib
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

_ML_ROOT = Path(__file__).resolve().parent.parent
if (_ML_ROOT / "ml_common" / "__init__.py").is_file() and str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from ml_common.async_fit import run_in_thread
from ml_common.learn_ids import ContractIdDedupe
from ml_common.names import META_FEATURE_NAMES
from ml_common.schema import canonical_schema_hash, request_schema_hash_ok, validate_feature_vector

from learn_runtime import (
    apply_label_scale,
    buffer_abs_mae,
    bundle_n_train,
    bundle_schema_ok,
    clamp_meta_edge,
    fit_regressor,
    is_tiny_online_bundle,
    load_learn_buffer,
    meta_retrain_floor,
    persist_regressor_bundle,
    save_learn_buffer,
    select_meta_bundle,
    should_retrain_meta,
)


logger = logging.getLogger("META")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
META_FEATURE_DIM = len(META_FEATURE_NAMES)
DEFAULT_FEATURE_NAMES: tuple[str, ...] = META_FEATURE_NAMES
SCHEMA_HASH = canonical_schema_hash(DEFAULT_FEATURE_NAMES)


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
    schema_hash: str = ""

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
    predicted_payoff_edge: float | None = None
    meta_applied: bool
    edge_expectancy: str
    model_version: str = ""
    source: str = ""


class LearnMetaRequest(BaseModel):
    feature_vector: list[float]
    target: float
    contract_id: str = ""
    symbol: str = ""
    schema_hash: str = ""


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
_learn_ids = ContractIdDedupe()
RETRAIN_MIN_N = int(os.getenv("META_RETRAIN_MIN_N", "32"))
MAX_BUFFER = int(os.getenv("META_MAX_BUFFER", "2000"))
BUFFER_PATH = MODELS_DIR / "meta_learn_buffer.pkl"


def _resolve_feature_names(bundle: dict[str, Any]) -> list[str]:
    stored = bundle.get("feature_names")
    if isinstance(stored, list) and stored:
        names = [str(name) for name in stored]
        if names != list(DEFAULT_FEATURE_NAMES):
            logger.warning(
                "feature_names do bundle divergem do schema canonico 23D; usando nomes do artefato",
            )
        return names
    return list(DEFAULT_FEATURE_NAMES)


def _build_feature_dataframe(bundle: dict[str, Any], feature_vector: list[float]) -> pl.DataFrame:
    names = _resolve_feature_names(bundle)
    row = validate_feature_vector(feature_vector, len(names))
    return pl.DataFrame({name: [row[idx]] for idx, name in enumerate(names)}).select(names)


def _model_source() -> str:
    name = str(_model_path.name) if _model_path is not None else ""
    if name.startswith("meta_online_") or bool((_model_bundle or {}).get("auto_learn_applied")):
        return "online"
    return "offline"


def _require_schema_hash(payload_hash: str) -> None:
    if not request_schema_hash_ok(payload_hash, SCHEMA_HASH):
        raise HTTPException(status_code=400, detail="schema_hash divergente")


def _model_version() -> str:
    if _model_path is not None:
        return _model_path.name
    return str((_model_bundle or {}).get("model_version") or "none")


def _try_load_bundle(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        logger.warning("Falha ao carregar modelo %s: %s", path, exc)
        return None, f"{path.name}: {exc}"
    if not isinstance(bundle, dict) or bundle.get("model") is None:
        return None, f"{path.name}: bundle sem chave model"
    model = bundle["model"]
    model_type = str(bundle.get("model_type") or "regressor")
    if model_type != "regressor":
        logger.warning("Artefato %s ignorado: model_type=%s", path.name, model_type)
        return None, f"{path.name}: model_type={model_type}"
    if not callable(getattr(model, "predict", None)):
        logger.warning("Artefato %s sem metodo predict", path.name)
        return None, f"{path.name}: metodo predict ausente"
    if not bundle_schema_ok(bundle, SCHEMA_HASH):
        logger.warning("schema_hash mismatch %s", path.name)
        return None, f"{path.name}: schema_hash mismatch"
    return bundle, None


def _load_model_bundle() -> dict[str, Any] | None:
    global _model_load_error, _model_path, _model_mtime, _n_loaded
    failures: list[str] = []
    if not MODELS_DIR.is_dir():
        _model_load_error = f"diretorio de modelos ausente: {MODELS_DIR}"
        return None
    candidates = list(MODELS_DIR.glob("*.pkl"))
    if not candidates:
        _model_load_error = f"nenhum artefato .pkl encontrado em {MODELS_DIR}"
        return None
    floor = meta_retrain_floor(int(RETRAIN_MIN_N))
    valid: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        if path.name == BUFFER_PATH.name:
            continue
        bundle, err = _try_load_bundle(path)
        if bundle is None:
            if err:
                failures.append(err)
            continue
        valid.append((path, bundle))
    if not valid:
        _model_load_error = "; ".join(failures) if failures else f"nenhum regressor valido em {MODELS_DIR}"
        return None
    mae = None
    online_items = [
        item
        for item in valid
        if str(item[0].name).startswith("meta_online_") or bool(item[1].get("auto_learn_applied"))
    ]
    online_items.sort(key=lambda item: bundle_n_train(item[1]), reverse=True)
    if online_items and _buffer_x:
        try:
            mae = buffer_abs_mae(online_items[0][1]["model"], _buffer_x, _buffer_y)
        except Exception:
            mae = None
    chosen = select_meta_bundle(valid, floor=floor, buffer_mae=mae)
    if chosen is None:
        _model_load_error = f"nenhum regressor valido em {MODELS_DIR}"
        return None
    path, bundle = chosen
    if is_tiny_online_bundle(path, bundle, floor=floor):
        offline = [item for item in valid if not is_tiny_online_bundle(item[0], item[1], floor=floor)]
        if offline:
            logger.info(
                "META: ignore tiny online n=%d version=%s floor=%d",
                bundle_n_train(bundle),
                path.name,
                floor,
            )
            path, bundle = offline[0]
    _model_load_error = None
    _model_path = path
    _model_mtime = float(path.stat().st_mtime)
    _n_loaded += 1
    feature_count = len(_resolve_feature_names(bundle))
    logger.info(
        "HEALTH schema=%s source=%s n=%d version=%s",
        SCHEMA_HASH[:12],
        "online" if str(path.name).startswith("meta_online_") else "offline",
        bundle_n_train(bundle),
        path.name,
    )
    logger.info("Modelo meta-regressor carregado: %s | feature_dim=%d", path.name, feature_count)
    return bundle


def _maybe_hot_reload() -> None:
    global _model_bundle, _model_mtime
    if not MODELS_DIR.is_dir():
        return
    candidates = [p for p in MODELS_DIR.glob("*.pkl") if p.name != BUFFER_PATH.name]
    if not candidates:
        return
    latest_mtime = max(float(p.stat().st_mtime) for p in candidates)
    if _model_bundle is not None and latest_mtime <= float(_model_mtime) + 1e-9:
        return
    bundle = _load_model_bundle()
    if bundle is not None:
        _model_bundle = bundle
        logger.info("Hot-reload meta: %s", _model_version())


def _regressor_unavailable_detail() -> str:
    if _model_load_error:
        return f"LGBMRegressor indisponivel: {_model_load_error}"
    return "LGBMRegressor indisponivel: modelo nao carregado no bootstrap"


@app.on_event("startup")
async def startup_load_model() -> None:
    global _model_bundle, _buffer_x, _buffer_y
    xs, ys = load_learn_buffer(BUFFER_PATH)
    _buffer_x = xs
    _buffer_y = ys
    _model_bundle = _load_model_bundle()
    if _model_bundle is None:
        logger.error(_regressor_unavailable_detail())


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
        "schema_hash": SCHEMA_HASH,
        "source": _model_source() if _model_bundle is not None else "",
        "degenerate": False,
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
    _require_schema_hash(payload.schema_hash)
    _maybe_hot_reload()
    bundle = _model_bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail=_regressor_unavailable_detail())
    model = bundle["model"]
    try:
        input_features_dataframe = _build_feature_dataframe(bundle, payload.feature_vector)
        raw_edge = model.predict(input_features_dataframe.to_numpy())[0]
        scaled = apply_label_scale(float(raw_edge), bundle)
        edge = clamp_meta_edge(scaled)
    except Exception as exc:
        logger.warning("Inferencia meta-regressor falhou: %s", exc)
        return MetaPredictResult(
            predicted_payoff_edge=None,
            meta_applied=False,
            edge_expectancy="LOSS_EXPECTED",
            model_version=_model_version(),
            source=_model_source(),
        )
    return MetaPredictResult(
        predicted_payoff_edge=edge,
        meta_applied=True,
        edge_expectancy=_classify_edge_expectancy(edge),
        model_version=_model_version(),
        source=_model_source(),
    )


@app.post("/v1/learn")
async def learn(payload: LearnMetaRequest) -> dict[str, Any]:
    global _model_bundle, _model_path, _model_mtime
    _require_schema_hash(payload.schema_hash)
    try:
        vector = validate_feature_vector(payload.feature_vector, META_FEATURE_DIM)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    retrained = False
    duplicate = False
    floor = meta_retrain_floor(int(RETRAIN_MIN_N))
    detail = "buffered"
    with _lock:
        if _learn_ids.seen_or_add(payload.contract_id):
            duplicate = True
            n = len(_buffer_y)
            do_fit = False
            xs = list(_buffer_x)
            ys = list(_buffer_y)
            names = _resolve_feature_names(_model_bundle) if _model_bundle is not None else list(DEFAULT_FEATURE_NAMES)
            detail = "duplicate_contract"
        else:
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
    if duplicate:
        pass
    elif not do_fit:
        detail = f"wait:{n}/{floor}"
    else:
        try:
            model = await run_in_thread(fit_regressor, xs, ys)
            path = persist_regressor_bundle(
                MODELS_DIR,
                model,
                n_train=len(ys),
                feature_names=names,
                feature_dim=META_FEATURE_DIM,
                schema_hash=SCHEMA_HASH,
            )
            with _lock:
                loaded = _load_model_bundle()
                if loaded is not None:
                    _model_bundle = loaded
            retrained = True
            detail = "ok" if _model_source() == "online" else "fit_kept_offline"
            logger.info(
                "LEARN schema=%s source=%s n=%d path=%s detail=%s",
                SCHEMA_HASH[:12],
                _model_source(),
                len(ys),
                path.name,
                detail,
            )
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
        "schema_hash": SCHEMA_HASH,
        "source": _model_source(),
        "degenerate": False,
        "duplicate": duplicate,
    }
