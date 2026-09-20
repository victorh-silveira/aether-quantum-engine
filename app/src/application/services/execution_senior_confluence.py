"""Decisao direcional de trader senior convertendo divergencias tecnicas em confluencia institucional."""

from __future__ import annotations

from typing import Any

from src.application.services.direction_loss_tracker import get_direction_loss_tracker
from src.application.services.execution_market_confluence import (
    _extract_edge_float,
    _extract_indicator_float,
)
from src.application.services.execution_price_action import _resolve_candle_ohlc
from src.application.services.execution_signal_skips import _closed_candle_dir
from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


def _check_marubozu_confluence(
    exec_dir: TradeDirection,
    ohlc: tuple[float, float, float, float],
) -> tuple[TradeDirection, str] | None:
    """Detecta expansao por marubozu oposto sem rejeicao."""
    open_px, high_px, low_px, close_px = ohlc
    range_px = high_px - low_px
    if range_px <= 1e-12:
        return None
    body_px = abs(close_px - open_px)
    if (body_px / range_px) < 0.80:
        return None
    if exec_dir == TradeDirection.CALL and close_px < open_px:
        lower_wick = (min(open_px, close_px) - low_px) / range_px
        if lower_wick < 0.15:
            return TradeDirection.PUT, "opposing_marubozu"
    elif exec_dir == TradeDirection.PUT and close_px > open_px:
        upper_wick = (high_px - max(open_px, close_px)) / range_px
        if upper_wick < 0.15:
            return TradeDirection.CALL, "opposing_marubozu"
    return None


def _check_wick_confluence(
    exec_dir: TradeDirection,
    ohlc: tuple[float, float, float, float],
) -> tuple[TradeDirection, str] | None:
    """Detecta rejeicao severa por pavio em extremidade."""
    open_px, high_px, low_px, close_px = ohlc
    range_px = high_px - low_px
    if range_px <= 1e-12:
        return None
    if exec_dir == TradeDirection.CALL:
        upper_wick = (high_px - max(open_px, close_px)) / range_px
        if upper_wick >= 0.45:
            return TradeDirection.PUT, "wick_rejection"
    elif exec_dir == TradeDirection.PUT:
        lower_wick = (min(open_px, close_px) - low_px) / range_px
        if lower_wick >= 0.45:
            return TradeDirection.CALL, "wick_rejection"
    return None


def _check_momentum_confluence(
    exec_dir: TradeDirection,
    metrics: dict[str, Any],
) -> tuple[TradeDirection, str] | None:
    """Detecta desequilibrio direcional severo de DI e RSI."""
    di_diff = _extract_indicator_float(metrics, "di_diff")
    rsi = _extract_indicator_float(metrics, "rsi")
    if di_diff is None or rsi is None:
        return None
    if exec_dir == TradeDirection.CALL and di_diff <= -0.15 and rsi <= 0.45:
        return TradeDirection.PUT, "directional_momentum"
    if exec_dir == TradeDirection.PUT and di_diff >= 0.15 and rsi >= 0.55:
        return TradeDirection.CALL, "directional_momentum"
    return None


def _check_tick_flow_confluence(
    exec_dir: TradeDirection,
    metrics: dict[str, Any],
) -> tuple[TradeDirection, str] | None:
    """Detecta aceleracao de micro-ticks oposta."""
    flow = metrics.get("flow_features")
    if not isinstance(flow, dict):
        return None
    try:
        vel = float(flow.get("price_velocity", flow.get("micro_tick_velocity", 0.0)) or 0.0)
        accel = float(flow.get("micro_tick_acceleration", flow.get("price_acceleration", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return None
    score = vel + 0.5 * accel
    if exec_dir == TradeDirection.CALL and score <= -1.2:
        return TradeDirection.PUT, "adverse_tick_flow"
    if exec_dir == TradeDirection.PUT and score >= 1.2:
        return TradeDirection.CALL, "adverse_tick_flow"
    return None


def _check_climactic_confluence(
    exec_dir: TradeDirection,
    ohlc: tuple[float, float, float, float],
    metrics: dict[str, Any],
) -> tuple[TradeDirection, str] | None:
    """Detecta exaustao climatica e reverte para mean reversion."""
    open_px, high_px, low_px, close_px = ohlc
    range_px = high_px - low_px
    atr = (
        _extract_indicator_float(metrics, "atr_norm")
        or _extract_indicator_float(metrics, "atr_abs")
        or _extract_indicator_float(metrics, "atr_raw")
        or _extract_indicator_float(metrics, "atr")
    )
    if atr is None or atr <= 0.5 or range_px <= 1e-12 or (range_px / atr) <= 2.5:
        return None
    rsi = _extract_indicator_float(metrics, "rsi")
    bb_b = _extract_indicator_float(metrics, "bb_pct_b")
    if (
        exec_dir == TradeDirection.CALL
        and close_px > open_px
        and ((rsi is not None and rsi > 0.75) or (bb_b is not None and bb_b > 1.05))
    ):
        return TradeDirection.PUT, "climactic_exhaustion"
    if (
        exec_dir == TradeDirection.PUT
        and close_px < open_px
        and ((rsi is not None and rsi < 0.25) or (bb_b is not None and bb_b < -0.05))
    ):
        return TradeDirection.CALL, "climactic_exhaustion"
    return None


def evaluate_senior_directional_decision(
    exec_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    symbol: str | None = None,
    exec_cfg: dict[str, Any] | None = None,
) -> tuple[TradeDirection, bool, str | None]:
    """Avalia confluencia tecnica de trader senior para operar a favor do fluxo dominante."""
    if exec_cfg is not None and not bool(exec_cfg.get("senior_confluence_flip", True)):
        return exec_dir, False, None
    if bool(metrics.get("loss_clf_flip")) or bool(metrics.get("anti_trend_lock_flip")):
        return exec_dir, False, None
    trend = str(metrics.get("trend_direction") or "").strip().upper()
    candle = _closed_candle_dir(metrics)
    edge = _extract_edge_float(metrics)
    if trend in _VALID and trend != exec_dir.name:
        if symbol:
            losses = get_direction_loss_tracker().consecutive_losses(str(symbol), exec_dir.name)
            if losses >= 1:
                return TradeDirection[trend], True, "anti_counter_trend_loss"
        if candle == trend:
            return TradeDirection[trend], True, "trend_candle_alignment"
        p_loss = metrics.get("loss_clf_p_loss")
        if p_loss is not None:
            try:
                if float(p_loss) >= 0.52:
                    return TradeDirection[trend], True, "loss_clf_macro_discord"
            except (TypeError, ValueError):
                pass
    elif edge >= 0.035:
        return exec_dir, False, None
    ohlc = _resolve_candle_ohlc(metrics, orch=orch, symbol=symbol)
    if ohlc is not None:
        maru = _check_marubozu_confluence(exec_dir, ohlc)
        if maru is not None:
            return maru[0], True, maru[1]
        wick = _check_wick_confluence(exec_dir, ohlc)
        if wick is not None:
            return wick[0], True, wick[1]
        climax = _check_climactic_confluence(exec_dir, ohlc, metrics)
        if climax is not None:
            return climax[0], True, climax[1]
    momo = _check_momentum_confluence(exec_dir, metrics)
    if momo is not None:
        return momo[0], True, momo[1]
    prev_bar = str(metrics.get("scale_micro_prev_bar_dir") or "").strip().upper()
    curr_bar = str(metrics.get("scale_micro_bar_dir") or candle or "").strip().upper()
    if prev_bar in _VALID and curr_bar == prev_bar and curr_bar != exec_dir.name:
        return TradeDirection[curr_bar], True, "two_bar_counter_trend"
    tick_flow = _check_tick_flow_confluence(exec_dir, metrics)
    if tick_flow is not None:
        return tick_flow[0], True, tick_flow[1]
    return exec_dir, False, None
