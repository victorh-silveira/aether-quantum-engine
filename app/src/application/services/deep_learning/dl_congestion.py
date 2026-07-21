"""Deteccao de congestionamento por squeeze de Bollinger."""

from __future__ import annotations

import numpy as np


def series_last(series: dict, key: str, default: float = 0.0) -> float:
    """Le ultimo valor de uma serie com fallback."""
    values = series.get(key)
    if values is None or len(values) == 0:
        return float(default)
    return float(values[-1])


def squeeze_congestion_active(
    prices,
    series: dict,
    *,
    bb_window: int,
    bb_std_mult: float,
    congestion: dict,
) -> bool:
    """Detecta congestionamento por squeeze de BB e ADX baixo."""
    window = max(2, int(bb_window))
    if len(prices) >= window:
        w_prices = prices[-window:]
        std_val = float(np.std(w_prices))
        mean_val = float(np.mean(w_prices))
        raw_bb_width = (2.0 * float(bb_std_mult) * std_val) / (mean_val + 1e-10)
    else:
        raw_bb_width = float(congestion["bb_width_max"]) + 0.01
    adx_val = series_last(series, "adx", 1.0)
    return (
        len(prices) >= int(congestion["min_bars"])
        and adx_val < float(congestion["adx_max"])
        and raw_bb_width < float(congestion["bb_width_max"])
    )
