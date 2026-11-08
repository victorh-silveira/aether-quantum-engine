"""Metricas puras de microestrutura de ticks em memoria."""

from __future__ import annotations

import math


def compute_buffer_microstructure(
    ticks: list[tuple[float, float]] | list[list[float]],
    *,
    window_seconds: int = 300,
) -> dict[str, float]:
    """Calcula microestrutura em tempo real a partir de buffer de ticks na memoria."""
    fallback: dict[str, float] = {
        "buy_tick_ratio": 0.5,
        "return_autocorr": 0.0,
        "final_momentum": 0.0,
        "realized_volatility": 0.0,
    }
    if not ticks or len(ticks) < 2:
        return fallback

    prices = [float(tick[1]) for tick in ticks]
    n_ticks = len(prices)
    deltas = [prices[i] - prices[i - 1] for i in range(1, n_ticks)]
    log_rets = [math.log(max(1e-12, prices[i]) / max(1e-12, prices[i - 1])) for i in range(1, n_ticks)]

    up_count = sum(1 for delta in deltas if delta > 0)
    buy_tick_ratio = float(up_count) / float(len(deltas)) if deltas else 0.5

    if len(log_rets) >= 2:
        autocorr_terms = [log_rets[i] * log_rets[i - 1] for i in range(1, len(log_rets))]
        return_autocorr = float(sum(autocorr_terms) / len(autocorr_terms))
    else:
        return_autocorr = 0.0

    tail_n = max(2, int(window_seconds * 0.15))
    final_momentum = float(prices[-1] - prices[-tail_n]) if n_ticks >= tail_n else float(prices[-1] - prices[0])

    sum_sq_ret = sum(ret * ret for ret in log_rets)
    realized_volatility = float(math.sqrt(sum_sq_ret))

    return {
        "buy_tick_ratio": buy_tick_ratio,
        "return_autocorr": return_autocorr,
        "final_momentum": final_momentum,
        "realized_volatility": realized_volatility,
    }
