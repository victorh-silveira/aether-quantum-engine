"""Valida a expectativa de Rise/Fall contra a cotacao do lado executado."""

from __future__ import annotations

import math
from typing import Any


def calculate_payout_breakeven_prob(payout_rate: float | None) -> float | None:
    """Calcula a probabilidade teorica de break-even (1 / (1 + rate))."""
    if payout_rate is None:
        return None
    try:
        rate = float(payout_rate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rate) or rate <= 0.0:
        return None
    return 1.0 / (1.0 + rate)


def _resolve_effective_side_probability(metrics: dict[str, Any], direction: str) -> float | None:
    """Resolve a probabilidade do lado considerando calibracao, loss_clf_flip e anti_trend_lock."""
    try:
        p_call = float(metrics["calibrated_prob"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(p_call) or not 0.0 <= p_call <= 1.0:
        return None
    if bool(metrics.get("loss_clf_flip")):
        try:
            raw_pe = metrics.get("loss_clf_p_eff") or metrics.get("loss_clf_p_loss")
            p_side = float(raw_pe) if raw_pe is not None else (p_call if direction == "CALL" else 1.0 - p_call)
        except (TypeError, ValueError):
            p_side = p_call if direction == "CALL" else 1.0 - p_call
    elif bool(metrics.get("anti_trend_lock_flip")) and metrics.get("conviction") is not None:
        try:
            p_side = float(metrics["conviction"])
        except (TypeError, ValueError):
            p_side = p_call if direction == "CALL" else 1.0 - p_call
    else:
        p_side = p_call if direction == "CALL" else 1.0 - p_call
    return p_side if math.isfinite(p_side) and 0.0 <= p_side <= 1.0 else None


def quoted_edge(
    metrics: dict[str, Any] | None,
    direction: str,
    payout_rate: float | None,
    *,
    probability_haircut: float = 0.0,
) -> float | None:
    """Retorna EV por unidade de stake; None significa evidencia insuficiente."""
    if not isinstance(metrics, dict) or direction not in {"CALL", "PUT"}:
        return None
    try:
        rate = float(payout_rate)  # type: ignore[arg-type]
        haircut = float(probability_haircut)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rate) or rate <= 0.0 or not math.isfinite(haircut) or not 0.0 <= haircut < 0.5:
        return None
    p_side = _resolve_effective_side_probability(metrics, direction)
    if p_side is None:
        return None
    return max(0.0, p_side - haircut) * (1.0 + rate) - 1.0


def is_quote_edge_acceptable(
    metrics: dict[str, Any] | None,
    direction: str,
    payout_rate: float | None,
    *,
    min_edge: float = 0.0,
    safety_margin: float = 0.0,
    probability_haircut: float = 0.0,
) -> tuple[bool, float | None, str]:
    """Verifica se a cotacao atende ao piso de EV e a margem de seguranca sobre o break-even."""
    edge = quoted_edge(metrics, direction, payout_rate, probability_haircut=probability_haircut)
    if edge is None:
        return False, None, "insufficient_evidence"
    if edge + 1e-12 < float(min_edge):
        return False, edge, "quote_edge_below_min"
    p_side = _resolve_effective_side_probability(metrics or {}, direction)
    p_be = calculate_payout_breakeven_prob(payout_rate)
    if p_side is not None and p_be is not None:
        margin_floor = float(safety_margin)
        if p_side + 1e-12 < p_be + margin_floor:
            return False, edge, "below_payout_breakeven_margin"
    return True, edge, "ok"


__all__ = [
    "calculate_payout_breakeven_prob",
    "is_quote_edge_acceptable",
    "quoted_edge",
]
