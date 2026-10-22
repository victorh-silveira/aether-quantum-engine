"""Modulo de analise institucional (VWAP Intraday, FVG e Killzones BRT)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite, sqrt

from indicator.domain import Candle

BRT_TIMEZONE = timezone(timedelta(hours=-3))


def session_vwap(bars: list[Candle]) -> tuple[float, float, float]:
    """Calcula VWAP causal da sessao diaria e bandas de 1.5 desvios padrao."""
    if not bars:
        return 0.0, 0.0, 0.0
    last_day = bars[-1].epoch // 86400
    session_bars = [b for b in bars if b.epoch // 86400 == last_day]
    if len(session_bars) < 2:
        session_bars = bars[-min(len(bars), 72) :]

    cum_vol = 0.0
    cum_pv = 0.0
    typical_prices: list[float] = []
    weights: list[float] = []

    for b in session_bars:
        tp = (b.high + b.low + b.close) / 3.0
        w = max(b.high - b.low, 1e-5)
        typical_prices.append(tp)
        weights.append(w)
        cum_pv += tp * w
        cum_vol += w

    vwap = cum_pv / cum_vol
    variance = sum(w * ((tp - vwap) ** 2) for tp, w in zip(typical_prices, weights, strict=True)) / cum_vol
    std_dev = sqrt(variance) if variance > 0.0 else 0.0
    upper_band = vwap + 1.5 * std_dev
    lower_band = vwap - 1.5 * std_dev
    return vwap, upper_band, lower_band


def detect_fvg_mitigation(bars: list[Candle], atr: float = 1.0) -> tuple[bool, bool]:
    """Detecta Fair Value Gaps recentes e valida se o recuo mitigou a ineficiencia."""
    if len(bars) < 5 or atr <= 0 or not isfinite(atr):
        return True, True

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

    if not bull_mitigated:
        for i in range(max(1, len(bars) - 5), len(bars) - 2):
            if bars[i].close - bars[i].open >= 0.8 * atr:
                mid = (bars[i].open + bars[i].close) / 2.0
                if pullback.low <= mid + 0.2 * atr:
                    bull_mitigated = True
                    break

    if not bear_mitigated:
        for i in range(max(1, len(bars) - 5), len(bars) - 2):
            if bars[i].open - bars[i].close >= 0.8 * atr:
                mid = (bars[i].open + bars[i].close) / 2.0
                if pullback.high >= mid - 0.2 * atr:
                    bear_mitigated = True
                    break

    return bull_mitigated, bear_mitigated


def is_within_killzone(epoch: int, optimal_hours: list[int] | None = None) -> bool:
    """Verifica se o timestamp da vela fechada pertence a Killzone em Horario de Brasilia."""
    if not optimal_hours:
        return True
    hour_brt = datetime.fromtimestamp(epoch, tz=BRT_TIMEZONE).hour
    return hour_brt in optimal_hours
