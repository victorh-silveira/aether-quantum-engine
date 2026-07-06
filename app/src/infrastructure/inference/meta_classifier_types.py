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
