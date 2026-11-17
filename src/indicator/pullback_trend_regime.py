"""Detector puro de regime adaptativo de tendencia M15 (R4/R6)."""

from __future__ import annotations

from indicator.domain import Candle


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
    spread = (fast[-1] - slow[-1]) / atr
    if abs(conf.close - fast[-1]) / atr > 1.50:
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
    up_count = sum(1 for bar in bars[-8:] if bar.close > bar.open)
    dn_count = sum(1 for bar in bars[-8:] if bar.close < bar.open)
    if up_count >= 6 and spread >= 1.0 and m_hist > 0.0 and conf.close > conf.open:
        return "CALL"
    if dn_count >= 6 and spread <= -1.0 and m_hist < 0.0 and conf.close < conf.open:
        return "PUT"
    return None
