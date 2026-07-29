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
    edge_expectancy: str


def classify_edge_expectancy_from_payoff(predicted_edge: float) -> str:
    """Classifica expectativa tabular a partir do edge continuo bruto."""
    edge = float(predicted_edge)
    if edge <= 0.0:
        return "LOSS_EXPECTED"
    if edge < 0.04:
        return "NO_EDGE_NEUTRAL"
    return "WIN_EXPECTED"


def parse_meta_predict_response(payload: object) -> MetaPredictResponse:
    """Extrai edge continuo da resposta HTTP do meta-regressor."""
    if not isinstance(payload, dict):
        raise TypeError("meta response must be object")
    if "predicted_payoff_edge" not in payload:
        raise KeyError("predicted_payoff_edge")
    edge = float(payload["predicted_payoff_edge"])
    applied = bool(payload.get("meta_applied", False))
    raw_expectancy = payload.get("edge_expectancy")
    if isinstance(raw_expectancy, str) and raw_expectancy.strip():
        expectancy = raw_expectancy.strip().upper()
    else:
        expectancy = classify_edge_expectancy_from_payoff(edge)
    return {
        "predicted_payoff_edge": edge,
        "meta_applied": applied,
        "edge_expectancy": expectancy,
    }


def resolve_predicted_edge(metrics: dict, payout: float = 0.95) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    prob = metrics.get("calibrated_prob", metrics.get("raw_prob", 0.5))
    if prob is None:
        return 0.0
    p = float(prob)
    p_win = max(p, 1.0 - p)
    return float((p_win * (1.0 + payout)) - 1.0)
