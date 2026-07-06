"""Tipos de payload para o cliente do meta-classificador tabular."""

from __future__ import annotations

from typing import TypedDict


class MetaPredictRequest(TypedDict):
    """Requisicao tipada enviada ao endpoint /v2/predict_meta."""

    symbol: str
    tcn_probability: float
    direction: str
    feature_vector: list[float]


class MetaPredictResponse(TypedDict):
    """Resposta tipada do meta-regressor com edge continuo de payoff."""

    predicted_payoff_edge: float
    meta_applied: bool


def parse_meta_predict_response(payload: object) -> MetaPredictResponse:
    """Extrai edge continuo da resposta HTTP do meta-regressor."""
    if not isinstance(payload, dict):
        raise TypeError("meta response must be object")
    if "predicted_payoff_edge" not in payload:
        raise KeyError("predicted_payoff_edge")
    edge = float(payload["predicted_payoff_edge"])
    applied = bool(payload.get("meta_applied", False))
    return {"predicted_payoff_edge": edge, "meta_applied": applied}
