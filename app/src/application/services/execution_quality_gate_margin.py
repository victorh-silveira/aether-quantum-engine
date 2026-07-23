"""Calculo e sincronizacao de direction_margin no quality gate."""

from __future__ import annotations

from typing import Any


def direction_margin_from_probability(call_probability: float, *, direction: str | None = None) -> float:
    """Calcula distancia da confianca lateral escolhida ao centro neutro 0.50."""
    call_score = max(0.0, min(1.0, float(call_probability)))
    side_probability = call_score
    if str(direction or "").upper() == "PUT":
        side_probability = 1.0 - call_score
    return abs(side_probability - 0.5)


def ensure_direction_margin(metrics: dict) -> float:
    """Garante direction_margin a partir da probabilidade calibrada ou bruta."""
    prob = metrics.get("calibrated_prob", metrics.get("raw_prob"))
    if prob is not None:
        direction = metrics.get("exec_direction") or metrics.get("resolved_direction") or metrics.get("dl_direction")
        margin = direction_margin_from_probability(
            float(prob),
            direction=str(direction) if direction is not None else None,
        )
    else:
        stored = metrics.get("direction_margin")
        margin = float(stored) if stored is not None else 0.0
    metrics["direction_margin"] = margin
    return margin


def sync_direction_margin(metrics: dict, *, direction: str) -> float:
    """Atualiza direction_margin a partir da probabilidade calibrada ou scores laterais."""
    prob = metrics.get("calibrated_prob", metrics.get("raw_prob"))
    if prob is not None:
        margin = direction_margin_from_probability(float(prob), direction=direction)
    else:
        margin = abs(float(metrics["direction_call_score"]) - float(metrics["direction_put_score"]))
    metrics["direction_margin"] = margin
    return margin


def stamp_edge_without_direction(
    metrics: dict[str, Any],
    *,
    margin_floor: float,
    score_factor: float = 0.85,
) -> None:
    """Marca Edge meta nao acionavel quando a margem TCN falha e comprime score."""
    metrics["edge_without_direction"] = True
    metrics["edge_without_direction_margin_floor"] = float(margin_floor)
    edge_raw = metrics.get("predicted_payoff_edge")
    if edge_raw is not None:
        metrics["edge_without_direction_edge"] = float(edge_raw)
    base = metrics.get("trade_score")
    if base is None:
        base = metrics.get("conviction")
    if base is None:
        return
    factor = max(0.0, min(1.0, float(score_factor)))
    compressed = float(base) * factor
    metrics["trade_score"] = compressed
    metrics["conviction"] = compressed
    metrics["edge_without_direction_penalty"] = max(0.0, float(base) - compressed)
