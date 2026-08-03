"""Calculo e sincronizacao de direction_margin no quality gate."""

from __future__ import annotations

import contextlib
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


def apply_quality_margin_floor_waivers(
    metrics: dict[str, Any],
    margin_floor: float,
    *,
    exec_cfg: dict[str, Any] | None,
) -> float:
    """Aplica waivers de piso de margem por senior trader ou Z-Score meta forte."""
    floor = float(margin_floor)
    senior = 0.0
    raw_senior = metrics.get("senior_trader_conviction")
    if raw_senior is not None:
        with contextlib.suppress(TypeError, ValueError):
            senior = float(raw_senior)
    if senior + 1e-12 >= 0.56:
        metrics["quality_margin_senior_waiver"] = True
        return 0.0
    from src.application.services.execution_quality_gate_config import (  # noqa: PLC0415
        resolve_quality_gate_config,
    )

    qg = resolve_quality_gate_config(exec_cfg)
    z_min = float(qg.get("min_meta_payoff_zscore", 0.0) or 0.0)
    z_raw = metrics.get("meta_payoff_edge_zscore")
    if z_raw is None:
        z_raw = metrics.get("edge_zscore")
    if z_min > 0.0 and z_raw is not None:
        with contextlib.suppress(TypeError, ValueError):
            if float(z_raw) + 1e-12 >= z_min:
                metrics["quality_margin_meta_z_waiver"] = True
                return 0.0
    return floor
