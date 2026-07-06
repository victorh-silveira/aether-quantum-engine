from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator


logger = logging.getLogger("META")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
FEATURE_DIM = 39
DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    *(f"feature_{index}" for index in range(34)),
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


class MetaPredictPayload(BaseModel):
    tcn_probability: float = Field(ge=0.0, le=1.0)
    direction: str
    feature_vector: list[float] = Field(min_length=FEATURE_DIM, max_length=FEATURE_DIM)
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


app = FastAPI(title="Aether Meta-Regressor", version="2.0.0")
_model_bundle: dict[str, Any] | None = None
_model_load_error: str | None = None


def _resolve_feature_names(bundle: dict[str, Any]) -> list[str]:
    stored = bundle.get("feature_names")
    if isinstance(stored, list) and len(stored) == FEATURE_DIM:
        names = [str(name) for name in stored]
        if names != list(DEFAULT_FEATURE_NAMES):
            logger.warning(
                "feature_names do bundle divergem do schema de treino; usando nomes do artefato",
            )
        return names
    return list(DEFAULT_FEATURE_NAMES)


def _build_feature_dataframe(bundle: dict[str, Any], feature_vector: list[float]) -> pd.DataFrame:
    names = _resolve_feature_names(bundle)
    if len(feature_vector) != FEATURE_DIM:
        raise ValueError(f"feature_vector deve ter {FEATURE_DIM} elementos, recebeu {len(feature_vector)}")
    if len(names) != FEATURE_DIM:
        raise ValueError(f"feature_names deve ter {FEATURE_DIM} colunas, recebeu {len(names)}")
    row = [float(value) for value in feature_vector]
    frame = pd.DataFrame([row], columns=names)
    return frame.loc[:, names]


def _load_model_bundle() -> dict[str, Any] | None:
    global _model_load_error
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
        logger.info("Modelo meta-regressor carregado: %s", path.name)
        return bundle
    _model_load_error = "; ".join(failures) if failures else f"nenhum regressor valido em {MODELS_DIR}"
    return None


def _regressor_unavailable_detail() -> str:
    if _model_load_error:
        return f"LGBMRegressor indisponivel: {_model_load_error}"
    return "LGBMRegressor indisponivel: modelo nao carregado no bootstrap"


@app.on_event("startup")
async def startup_load_model() -> None:
    global _model_bundle
    _model_bundle = _load_model_bundle()
    if _model_bundle is None:
        logger.error(_regressor_unavailable_detail())


@app.get("/health")
async def health() -> dict[str, bool | str]:
    return {
        "ready": _model_bundle is not None,
        "model_loaded": _model_bundle is not None,
        "load_error": _model_load_error or "",
    }


@app.post("/v2/predict_meta", response_model=MetaPredictResult)
async def predict_meta(payload: MetaPredictPayload) -> MetaPredictResult:
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
        return MetaPredictResult(predicted_payoff_edge=0.0, meta_applied=False)
    return MetaPredictResult(predicted_payoff_edge=edge, meta_applied=True)
