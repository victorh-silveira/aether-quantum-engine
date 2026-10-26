"""Modulo de analise institucional (referencia OHLC, FVG e killzones BRT)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite, sqrt

from indicator.domain import Candle

BRT_TIMEZONE = timezone(timedelta(hours=-3))
DEFAULT_KILLZONES_BRT = [3, 4, 5, 6, 7, 10, 11, 12, 14, 15]


def session_price_reference(bars: list[Candle]) -> tuple[float, float, float]:
    """Calcula referencia temporal OHLC da sessao UTC, sem alegar volume inexistente."""
    if not bars:
        return 0.0, 0.0, 0.0
    last_day = bars[-1].epoch // 86400
    session_bars = [b for b in bars if b.epoch // 86400 == last_day]
    if len(session_bars) < 2:
        return 0.0, 0.0, 0.0

    sample_count = 0.0
    price_sum = 0.0
    typical_prices: list[float] = []
    weights: list[float] = []

    for b in session_bars:
        tp = (b.high + b.low + b.close) / 3.0
        w = 1.0
        typical_prices.append(tp)
        weights.append(w)
        price_sum += tp * w
        sample_count += w

    reference = price_sum / sample_count
    variance = sum(w * ((tp - reference) ** 2) for tp, w in zip(typical_prices, weights, strict=True)) / sample_count
    std_dev = sqrt(variance) if variance > 0.0 else 0.0
    upper_band = reference + 1.5 * std_dev
    lower_band = reference - 1.5 * std_dev
    return reference, upper_band, lower_band


def detect_fvg_mitigation(bars: list[Candle], atr: float = 1.0) -> tuple[bool, bool]:
    """Detecta Fair Value Gaps recentes e valida se o recuo mitigou a ineficiencia."""
    if len(bars) < 5 or atr <= 0 or not isfinite(atr):
        return False, False

    pullback = bars[-2]
    bull_mitigated = False
    bear_mitigated = False

    for i in range(max(2, len(bars) - 6), len(bars) - 2):
        bar_prev2 = bars[i - 2]
        bar_curr = bars[i]

        if bar_curr.low > bar_prev2.high:
            fvg_top = bar_curr.low
            fvg_bottom = bar_prev2.high
            if pullback.low <= fvg_top + 0.10 * atr and pullback.low >= fvg_bottom - 0.20 * atr:
                bull_mitigated = True

        if bar_curr.high < bar_prev2.low:
            fvg_bottom = bar_curr.high
            fvg_top = bar_prev2.low
            if pullback.high >= fvg_bottom - 0.10 * atr and pullback.high <= fvg_top + 0.20 * atr:
                bear_mitigated = True

    return bull_mitigated, bear_mitigated


def is_within_killzone(epoch: int, optimal_hours: list[int] | None = None) -> bool:
    """Verifica se o timestamp da vela fechada pertence a Killzone em Horario de Brasilia."""
    if optimal_hours is None:
        optimal_hours = DEFAULT_KILLZONES_BRT
    if not optimal_hours:
        return False
    hour_brt = datetime.fromtimestamp(epoch, tz=BRT_TIMEZONE).hour
    return hour_brt in optimal_hours
