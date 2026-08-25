"""Helpers de indicadores tecnicos (EMA, RSI) para o gate anti-loss."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.domain.models.trade import TradeDirection


def calc_ema_series(series: np.ndarray, period: int) -> np.ndarray | None:
    """Calcula a serie completa da EMA exponencial."""
    if len(series) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    out = np.zeros(len(series), dtype=np.float64)
    out[0] = float(series[0])
    for i in range(1, len(series)):
        out[i] = alpha * float(series[i]) + (1.0 - alpha) * out[i - 1]
    return out


def calc_ema(series: np.ndarray, period: int) -> float | None:
    """Calcula o ultimo valor da EMA exponencial para a serie."""
    s = calc_ema_series(series, period)
    return float(s[-1]) if s is not None and len(s) > 0 else None


def check_mini_ema_trend_and_slope(
    orch: Any | None,
    symbol: str | None,
    side: TradeDirection,
    metrics: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Valida alinhamento e slope da EMA21 no timeframe M5."""
    if orch is None or not symbol:
        return True, None
    stream = getattr(orch, "stream", None)
    if stream is None or not hasattr(stream, "get_mini_numpy_series"):
        return True, None
    closes = stream.get_mini_numpy_series(str(symbol), "close")
    if len(closes) < 9:
        return True, None
    ema9 = float(calc_ema(closes, 9))
    last_close = float(closes[-1])
    ema21_series = calc_ema_series(closes, 21) if len(closes) >= 21 else None
    tol = 0.50
    if metrics is not None:
        atr_val = metrics.get("atr")
        if atr_val is not None and float(atr_val) > 0.0:
            tol = max(tol, float(atr_val) * 0.4)
    if side == TradeDirection.CALL:
        if last_close < ema9 - tol:
            return False, "anti_loss_ema_trend"
        if ema21_series is not None and len(ema21_series) >= 3:
            ema21_last = float(ema21_series[-1])
            if ema21_last < float(ema21_series[-3]) - 0.10:
                return False, "anti_loss_ema_slope"
    elif side == TradeDirection.PUT:
        if last_close > ema9 + tol:
            return False, "anti_loss_ema_trend"
        if ema21_series is not None and len(ema21_series) >= 3:
            ema21_last = float(ema21_series[-1])
            if ema21_last > float(ema21_series[-3]) + 0.10:
                return False, "anti_loss_ema_slope"
    return True, None


def check_rsi_filter(
    metrics: dict[str, Any],
    side: TradeDirection,
) -> bool:
    """True se RSI intradiario for valido: CALL >= 0.32 e PUT <= 0.68."""
    indicators = metrics.get("indicators") or {}
    micro = metrics.get("micro_indicators") or {}
    rsi_val = indicators.get("rsi")
    if rsi_val is None and isinstance(micro, dict):
        rsi_val = micro.get("rsi")
    if rsi_val is None:
        return True
    try:
        rsi = float(rsi_val)
        if rsi > 1.0:
            rsi = rsi / 100.0
    except (TypeError, ValueError):
        return True
    call_blocked = side == TradeDirection.CALL and rsi < 0.32
    put_blocked = side == TradeDirection.PUT and rsi > 0.68
    return not (call_blocked or put_blocked)
