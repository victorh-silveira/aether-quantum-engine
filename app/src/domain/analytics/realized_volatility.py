"""Calculo de volatilidade realizada (RV) de alta frequencia para micro-ticks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def compute_realized_volatility(prices: Sequence[float]) -> float:
    """Calcula a volatilidade realizada como a raiz da soma dos retornos logaritmicos quadraticos."""
    if len(prices) < 2:
        return 0.0
    sum_sq = 0.0
    valid_count = 0
    prev = float(prices[0])
    for curr_val in prices[1:]:
        curr = float(curr_val)
        if prev > 0.0 and curr > 0.0 and math.isfinite(prev) and math.isfinite(curr):
            ret = math.log(curr / prev)
            sum_sq += ret * ret
            valid_count += 1
        prev = curr
    if valid_count == 0:
        return 0.0
    return float(math.sqrt(sum_sq))


def compute_micro_volatility_ratio(
    prices_short: Sequence[float],
    prices_medium: Sequence[float],
    *,
    time_ratio: float = 0.20,
) -> float:
    """Calcula a razao de volatilidade realizada entre janela curta e janela media normalizada pelo tempo."""
    rv_short = compute_realized_volatility(prices_short)
    rv_medium = compute_realized_volatility(prices_medium)
    if rv_medium <= 1e-12:
        return 1.0
    expected_factor = math.sqrt(max(1e-4, float(time_ratio)))
    expected_short = rv_medium * expected_factor
    if expected_short <= 1e-12:
        return 1.0
    ratio = rv_short / expected_short
    return float(max(0.0, min(10.0, ratio)))


def diagnose_micro_volatility_regime(
    tick_records: Sequence[tuple[float, float]],
    *,
    current_time: float | None = None,
    window_short_seconds: float = 60.0,
    window_medium_seconds: float = 300.0,
) -> dict[str, Any]:
    """Diagnostica o regime microestrutural de volatilidade a partir de tuplas (timestamp, preco)."""
    if not tick_records:
        return {
            "rv_short": 0.0,
            "rv_medium": 0.0,
            "vol_ratio": 1.0,
            "regime": "normal",
            "doji_risk": False,
            "tick_count_short": 0,
            "tick_count_medium": 0,
        }
    now_ts = float(current_time if current_time is not None else tick_records[-1][0])
    short_cutoff = now_ts - float(window_short_seconds)
    medium_cutoff = now_ts - float(window_medium_seconds)
    prices_short: list[float] = []
    prices_medium: list[float] = []
    for ts, px in tick_records:
        t_val, p_val = float(ts), float(px)
        if t_val >= medium_cutoff:
            prices_medium.append(p_val)
            if t_val >= short_cutoff:
                prices_short.append(p_val)
    rv_short = compute_realized_volatility(prices_short)
    rv_medium = compute_realized_volatility(prices_medium)
    t_ratio = float(window_short_seconds) / float(max(1.0, window_medium_seconds))
    ratio = compute_micro_volatility_ratio(prices_short, prices_medium, time_ratio=t_ratio)
    if ratio <= 0.40:
        regime = "compression"
        doji_risk = True
    elif ratio >= 1.80:
        regime = "explosion"
        doji_risk = False
    else:
        regime = "normal"
        doji_risk = False
    return {
        "rv_short": rv_short,
        "rv_medium": rv_medium,
        "vol_ratio": ratio,
        "regime": regime,
        "doji_risk": doji_risk,
        "tick_count_short": len(prices_short),
        "tick_count_medium": len(prices_medium),
    }


__all__ = [
    "compute_micro_volatility_ratio",
    "compute_realized_volatility",
    "diagnose_micro_volatility_regime",
]
