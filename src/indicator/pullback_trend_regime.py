"""Detector puro de regime adaptativo de tendencia M15 (R4/R6)."""

from __future__ import annotations

from src.indicator.domain import Candle


def evaluate_trend_regime(
    bars: list[Candle],
    fast: list[float],
    slow: list[float],
    atr: float,
    *,
    adx: float,
    pdi: float,
    mdi: float,
    m_hist: float,
    hurst: float,
    mode: str = "r6",
) -> str | None:
    """Avalia entrada a favor da tendencia apenas sob regime persistente (R4 ou R6)."""
    norm = str(mode or "off").strip().lower()
    if norm not in {"r4", "r6"} or atr <= 0 or len(bars) < 8 or not fast or not slow:
        return None
    conf = bars[-1]
    body = abs(conf.close - conf.open)
    candle_range = max(1e-10, conf.high - conf.low)
    if body < 0.10 * atr or body < 0.15 * candle_range:
        return None
    spread = (fast[-1] - slow[-1]) / atr
    max_dist = 2.50 if (adx >= 35.0 and hurst >= 0.62) else 1.50
    if abs(conf.close - fast[-1]) / atr > max_dist:
        return None
    min_adx = 35.0 if norm == "r4" else 30.0
    min_hurst = 0.62 if norm == "r4" else 0.60
    if adx < min_adx or hurst < min_hurst:
        return None
    if norm == "r4":
        if spread >= 1.0 and pdi > mdi and m_hist > 0.0:
            return "CALL"
        if spread <= -1.0 and mdi > pdi and m_hist < 0.0:
            return "PUT"
        return None
    upper_wick = conf.high - max(conf.open, conf.close)
    lower_wick = min(conf.open, conf.close) - conf.low
    up_count = sum(1 for bar in bars[-8:] if bar.close > bar.open)
    dn_count = sum(1 for bar in bars[-8:] if bar.close < bar.open)
    if (
        up_count >= 5
        and spread >= 0.80
        and pdi > mdi
        and m_hist > 0.0
        and conf.close > conf.open
        and upper_wick < 0.50 * candle_range
    ):
        return "CALL"
    if (
        dn_count >= 5
        and spread <= -0.80
        and mdi > pdi
        and m_hist < 0.0
        and conf.close < conf.open
        and lower_wick < 0.50 * candle_range
    ):
        return "PUT"
    return None
