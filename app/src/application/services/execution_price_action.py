"""Salvaguardas de price action e microestrutura: rejeicao por pavio, climax e fluxo adverso."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_market_confluence import (
    _extract_edge_float,
    _extract_indicator_float,
)
from src.application.services.execution_signal_skips import _mark_skip
from src.application.services.market_audit_candle import last_closed_micro_candle
from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection


def _resolve_candle_ohlc(
    metrics: dict[str, Any],
    orch: Any | None = None,
    symbol: str | None = None,
) -> tuple[float, float, float, float] | None:
    """Extrai (open, high, low, close) da ultima vela fechada via metrics ou stream."""
    raw = metrics.get("closed_candle_ohlc")
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            return float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
        except (TypeError, ValueError):
            pass
    if orch is not None and symbol:
        stream = getattr(orch, "stream", None)
        candle = last_closed_micro_candle(stream, str(symbol))
        if isinstance(candle, Candle):
            return float(candle.open), float(candle.high), float(candle.low), float(candle.close)
    return None


def should_skip_wick_rejection(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    orch: Any | None = None,
    symbol: str | None = None,
    force: bool = False,
) -> bool:
    """Bloqueia ordens contra longa sombra de rejeicao na vela fechada."""
    if force or not bool((exec_cfg or {}).get("skip_wick_rejection", False)):
        return False
    if bool(metrics.get("loss_clf_flip")) or bool(metrics.get("anti_trend_lock_flip")):
        return False
    ohlc = _resolve_candle_ohlc(metrics, orch=orch, symbol=symbol)
    if ohlc is None:
        return False
    open_px, high_px, low_px, close_px = ohlc
    range_px = high_px - low_px
    if range_px <= 1e-12:
        return False
    edge = _extract_edge_float(metrics)
    if edge >= 0.035:
        return False
    trend_raw = str(metrics.get("trend_direction") or "").strip().upper()
    if trend_raw == exec_dir.name and edge >= 0.020:
        return False
    if exec_dir == TradeDirection.CALL:
        upper_wick = high_px - max(open_px, close_px)
        ratio = upper_wick / range_px
        if ratio >= 0.45:
            _mark_skip(metrics, "wick_rejection_call", upper_wick_ratio=float(ratio))
            return True
    elif exec_dir == TradeDirection.PUT:
        lower_wick = min(open_px, close_px) - low_px
        ratio = lower_wick / range_px
        if ratio >= 0.45:
            _mark_skip(metrics, "wick_rejection_put", lower_wick_ratio=float(ratio))
            return True
    return False


def should_skip_climactic_blowoff(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    orch: Any | None = None,
    symbol: str | None = None,
    force: bool = False,
) -> bool:
    """Bloqueia continuidade imediata apos vela anomala de exaustao de volatilidade."""
    if force or not bool((exec_cfg or {}).get("skip_climactic_blowoff", False)):
        return False
    if bool(metrics.get("loss_clf_flip")) or bool(metrics.get("anti_trend_lock_flip")):
        return False
    ohlc = _resolve_candle_ohlc(metrics, orch=orch, symbol=symbol)
    if ohlc is None:
        return False
    open_px, high_px, low_px, close_px = ohlc
    range_px = high_px - low_px
    atr = (
        _extract_indicator_float(metrics, "atr_raw")
        or _extract_indicator_float(metrics, "atr_abs")
        or _extract_indicator_float(metrics, "atr")
        or _extract_indicator_float(metrics, "atr_norm")
    )
    if atr is None or range_px <= 1e-12:
        return False
    if atr <= 0.5:
        atr = max(1.0, open_px * 0.001)
    if (range_px / atr) <= 2.5:
        return False
    edge = _extract_edge_float(metrics)
    if edge >= 0.035:
        return False
    is_bullish = close_px > open_px
    is_bearish = close_px < open_px
    if exec_dir == TradeDirection.CALL and is_bullish:
        _mark_skip(metrics, "climactic_blowoff_call", range_atr_ratio=float(range_px / atr))
        return True
    if exec_dir == TradeDirection.PUT and is_bearish:
        _mark_skip(metrics, "climactic_blowoff_put", range_atr_ratio=float(range_px / atr))
        return True
    return False


def should_skip_adverse_tick_flow(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    force: bool = False,
) -> bool:
    """Bloqueia ordens quando o fluxo de micro-ticks e a aceleracao sao contrarios."""
    if force or not bool((exec_cfg or {}).get("skip_adverse_tick_flow", False)):
        return False
    if bool(metrics.get("loss_clf_flip")) or bool(metrics.get("anti_trend_lock_flip")):
        return False
    flow = metrics.get("flow_features")
    if not isinstance(flow, dict):
        return False
    try:
        vel = float(flow.get("price_velocity", flow.get("micro_tick_velocity", 0.0)) or 0.0)
    except (TypeError, ValueError):
        vel = 0.0
    try:
        accel = float(flow.get("micro_tick_acceleration", flow.get("price_acceleration", 0.0)) or 0.0)
    except (TypeError, ValueError):
        accel = 0.0
    flow_score = vel + 0.5 * accel
    edge = _extract_edge_float(metrics)
    if edge >= 0.050:
        return False
    if exec_dir == TradeDirection.CALL and flow_score <= -1.2:
        _mark_skip(metrics, "adverse_tick_flow_call", flow_score=float(flow_score))
        return True
    if exec_dir == TradeDirection.PUT and flow_score >= 1.2:
        _mark_skip(metrics, "adverse_tick_flow_put", flow_score=float(flow_score))
        return True
    return False


def should_skip_opposing_marubozu_flow(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    orch: Any | None = None,
    symbol: str | None = None,
    force: bool = False,
) -> bool:
    """Bloqueia ordens contra vela Marubozu de forca oposta sem absorcao institucional."""
    if force or not bool((exec_cfg or {}).get("skip_opposing_marubozu", False)):
        return False
    if bool(metrics.get("loss_clf_flip")) or bool(metrics.get("anti_trend_lock_flip")):
        return False
    ohlc = _resolve_candle_ohlc(metrics, orch=orch, symbol=symbol)
    if ohlc is None:
        return False
    open_px, high_px, low_px, close_px = ohlc
    range_px = high_px - low_px
    if range_px <= 1e-12:
        return False
    body_px = abs(close_px - open_px)
    if (body_px / range_px) < 0.80:
        return False
    edge = _extract_edge_float(metrics)
    if edge >= 0.035:
        return False
    trend = str(metrics.get("trend_direction") or "").strip().upper()
    if trend == exec_dir.name and edge >= 0.020:
        return False
    if exec_dir == TradeDirection.CALL and close_px < open_px:
        lower_wick = (min(open_px, close_px) - low_px) / range_px
        if lower_wick < 0.15:
            _mark_skip(metrics, "opposing_bearish_marubozu", body_ratio=float(body_px / range_px))
            return True
    elif exec_dir == TradeDirection.PUT and close_px > open_px:
        upper_wick = (high_px - max(open_px, close_px)) / range_px
        if upper_wick < 0.15:
            _mark_skip(metrics, "opposing_bullish_marubozu", body_ratio=float(body_px / range_px))
            return True
    return False
