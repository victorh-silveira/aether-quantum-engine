"""Helpers de indicadores tecnicos (EMA, RSI) para o gate anti-loss."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.domain.models.trade import TradeDirection


_ema_cache: dict[tuple[int, int, int], np.ndarray] = {}
_ema_cache_cycle: int | None = None


def _ema_cache_key(series: np.ndarray, period: int) -> tuple[int, int, int]:
    """Chave de cache deterministica para a serie e periodo da EMA."""
    return (id(series), len(series), period)


def invalidate_ema_cache(cycle_id: int | None = None) -> None:
    """Limpa o cache de EMA; chamar no inicio de cada ciclo do orquestrador."""
    global _ema_cache_cycle  # noqa: PLW0603
    _ema_cache.clear()
    _ema_cache_cycle = cycle_id


def _compute_ema_array(series: np.ndarray, period: int) -> np.ndarray:
    """Calcula a serie EMA exponencial sem cache."""
    alpha = 2.0 / (period + 1.0)
    out = np.zeros(len(series), dtype=np.float64)
    out[0] = float(series[0])
    for i in range(1, len(series)):
        out[i] = alpha * float(series[i]) + (1.0 - alpha) * out[i - 1]
    return out


def calc_ema_series(series: np.ndarray, period: int) -> np.ndarray | None:
    """Calcula a serie completa da EMA exponencial com cache por ciclo."""
    if len(series) < period:
        return None
    key = _ema_cache_key(series, period)
    cached = _ema_cache.get(key)
    if cached is not None and len(cached) == len(series):
        return cached
    result = _compute_ema_array(series, period)
    _ema_cache[key] = result
    return result


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
    if closes is None or len(closes) < 9:
        return True, None
    last_close = float(closes[-1])
    ema9 = float(calc_ema(closes, 9))
    ema21_series = calc_ema_series(closes, 21) if len(closes) >= 21 else None
    base_tol = max(0.50, last_close * 0.001)
    base_slope_tol = max(0.10, last_close * 0.0002)
    tol = base_tol
    slope_tol = base_slope_tol
    if metrics is not None:
        atr_val = metrics.get("atr")
        if atr_val is not None and float(atr_val) > 0.0:
            tol = max(base_tol, float(atr_val) * 0.4)
            slope_tol = max(base_slope_tol, float(atr_val) * 0.15)
    ema9_series = calc_ema_series(closes, 9)
    ema9_slope_tol = slope_tol * 0.6
    if side == TradeDirection.CALL:
        if last_close < ema9 - tol:
            return False, "anti_loss_ema_trend"
        if (
            ema9_series is not None
            and len(ema9_series) >= 2
            and float(ema9_series[-1]) < float(ema9_series[-2]) - ema9_slope_tol
        ):
            return False, "anti_loss_ema_slope"
        if ema21_series is not None and len(ema21_series) >= 2:
            ema21_last = float(ema21_series[-1])
            if ema21_last < float(ema21_series[-2]) - slope_tol:
                return False, "anti_loss_ema_slope"
    elif side == TradeDirection.PUT:
        if last_close > ema9 + tol:
            return False, "anti_loss_ema_trend"
        if (
            ema9_series is not None
            and len(ema9_series) >= 2
            and float(ema9_series[-1]) > float(ema9_series[-2]) + ema9_slope_tol
        ):
            return False, "anti_loss_ema_slope"
        if ema21_series is not None and len(ema21_series) >= 2:
            ema21_last = float(ema21_series[-1])
            if ema21_last > float(ema21_series[-2]) + slope_tol:
                return False, "anti_loss_ema_slope"
    return True, None


def check_rsi_filter(
    metrics: dict[str, Any],
    side: TradeDirection,
    rsi_min: float = 0.30,
    rsi_max: float = 0.70,
) -> bool:
    """True se RSI intradiario for valido dentro dos limites operacionais."""
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
    call_blocked = side == TradeDirection.CALL and rsi < rsi_min
    put_blocked = side == TradeDirection.PUT and rsi > rsi_max
    return not (call_blocked or put_blocked)


_SOFT_MICRO = frozenset(
    {
        "anti_loss_ema_trend",
        "anti_loss_ema_slope",
        "anti_loss_rsi_momentum",
        "anti_loss_rsi_trend",
        "live_discord_weak",
        "live_confirm_weak",
        "live_weak_candle",
        "live_no_candle",
    }
)


def finalize_anti_loss_decision(out: dict[str, Any], *, cfg: dict[str, Any], reason: str) -> dict[str, Any]:
    """Marca decisao; EMA/RSI/confirm/weak soft; seed respeita hard_skip."""
    out["active"], out["reason"] = True, reason
    if reason in _SOFT_MICRO or not bool(cfg.get("anti_loss_hard_skip", True)):
        out["soft"], out["soft_mult"] = True, float(cfg.get("anti_loss_soft_kelly_mult", 0.55))
        return out
    out["skip"] = True
    return out
