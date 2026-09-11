"""Tipos de payload do loss-classifier HTTP."""

from __future__ import annotations

from typing import TypedDict


BUFFER_N_ABSENT = -1


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
    bootstrap: bool
    collapsed: bool
    buffer_n: int
    bootstrap_exit_n: int


def parse_loss_predict_response(payload: object) -> LossPredictResponse:
    """Extrai campos obrigatorios da resposta HTTP."""
    if not isinstance(payload, dict):
        raise TypeError("loss response must be object")
    buffer_n = int(payload.get("buffer_n") or 0) if "buffer_n" in payload else BUFFER_N_ABSENT
    return {
        "p_loss": float(payload.get("p_loss", 0.5)),
        "veto": bool(payload.get("veto", False)),
        "auto_learn_applied": bool(payload.get("auto_learn_applied", False)),
        "model_version": str(payload.get("model_version") or "none"),
        "n_train": int(payload.get("n_train") or 0),
        "veto_ready": bool(payload.get("veto_ready", False)),
        "bootstrap": bool(payload.get("bootstrap", False)),
        "collapsed": bool(payload.get("collapsed", False)),
        "buffer_n": buffer_n,
        "bootstrap_exit_n": max(1, int(payload.get("bootstrap_exit_n") or 4)),
    }
