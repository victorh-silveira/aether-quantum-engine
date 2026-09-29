"""Valida a expectativa de Rise/Fall contra a cotacao do lado executado."""

from __future__ import annotations

import math
from typing import Any


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
        p_call = float(metrics["calibrated_prob"])
        rate = float(payout_rate)
        haircut = float(probability_haircut)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(value) for value in (p_call, rate, haircut))
        or not 0.0 <= p_call <= 1.0
        or rate <= 0.0
        or not 0.0 <= haircut < 0.5
    ):
        return None
    p_side = p_call if direction == "CALL" else 1.0 - p_call
    return max(0.0, p_side - haircut) * (1.0 + rate) - 1.0
