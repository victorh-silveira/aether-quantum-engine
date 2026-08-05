"""Tipos de payload do loss-classifier HTTP."""

from __future__ import annotations

from typing import TypedDict


class LossPredictRequest(TypedDict):
    """Requisicao /v1/predict_loss."""

    feature_vector: list[float]
    symbol: str
    direction: str
    veto_p_loss_floor: float


class LossPredictResponse(TypedDict):
    """Resposta tipada do loss-classifier."""

    p_loss: float
    veto: bool
    auto_learn_applied: bool
    model_version: str
    n_train: int
    veto_ready: bool


def parse_loss_predict_response(payload: object) -> LossPredictResponse:
    """Extrai campos obrigatorios da resposta HTTP."""
    if not isinstance(payload, dict):
        raise TypeError("loss response must be object")
    return {
        "p_loss": float(payload.get("p_loss", 0.5)),
        "veto": bool(payload.get("veto", False)),
        "auto_learn_applied": bool(payload.get("auto_learn_applied", False)),
        "model_version": str(payload.get("model_version") or "none"),
        "n_train": int(payload.get("n_train") or 0),
        "veto_ready": bool(payload.get("veto_ready", False)),
    }
