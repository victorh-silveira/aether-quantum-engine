"""Estimativa bayesiana de win rate para Kelly."""

from __future__ import annotations

import math
from typing import Any


def _shrink_toward_half(p: float, *, weight: float) -> float:
    """Aproxima p de 0.50 com peso weight."""
    return (1.0 - weight) * p + weight * 0.50


def _apply_live_quality_shrink(p: float, bag: dict[str, Any]) -> float:
    """Aplica shrink por Brier/ECE elevados nas metricas live."""
    out = p
    try:
        if bag.get("live_brier") is not None and float(bag["live_brier"]) > 0.24:
            out = _shrink_toward_half(out, weight=0.5)
    except (TypeError, ValueError):
        pass
    try:
        if bag.get("live_ece") is not None and float(bag["live_ece"]) > 0.10:
            out = _shrink_toward_half(out, weight=0.5)
    except (TypeError, ValueError):
        pass
    return out


def _live_blend_weight(live_n: int, bag: dict[str, Any]) -> float:
    """Calcula peso de blend live_wr com reducao por qualidade."""
    w = min(0.55, float(live_n) / 64.0)
    try:
        if bag.get("live_brier") is not None and float(bag["live_brier"]) > 0.24:
            w *= 0.5
    except (TypeError, ValueError):
        pass
    try:
        if bag.get("live_ece") is not None and float(bag["live_ece"]) > 0.10:
            w *= 0.5
    except (TypeError, ValueError):
        pass
    return w


def bayesian_win_rate(
    conviction: float,
    *,
    rolling_wr: float | None = None,
    rolling_n: int = 0,
    metrics: dict[str, Any] | None = None,
    dynamic_min_samples: int = 6,
) -> float:
    """Estima p Kelly com prior de conviccao, live_wr e shrink por Brier/ECE/Z."""
    prior = max(0.45, min(0.70, float(conviction)))
    bag = metrics if isinstance(metrics, dict) else {}
    live_n = int(bag.get("live_n", 0) or 0)
    live_wr = bag.get("live_wr")
    p = prior
    if live_n >= 20 and live_wr is not None:
        try:
            live_p = float(live_wr)
        except (TypeError, ValueError):
            live_p = prior
        w = _live_blend_weight(live_n, bag)
        p = (1.0 - w) * prior + w * live_p
        p = _apply_live_quality_shrink(p, bag)
    elif rolling_wr is not None and int(rolling_n) >= int(dynamic_min_samples):
        p = prior * 0.7 + float(rolling_wr) * 0.3
    z_raw = bag.get("meta_payoff_edge_zscore", bag.get("edge_zscore"))
    if z_raw is not None:
        try:
            adj = max(-0.03, min(0.03, 0.02 * math.tanh(float(z_raw))))
            p += adj
        except (TypeError, ValueError):
            pass
    return max(0.40, min(0.75, float(p)))
