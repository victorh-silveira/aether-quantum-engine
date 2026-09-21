"""Decisao direcional de trader senior convertendo divergencias tecnicas em confluencia institucional."""

from __future__ import annotations

from typing import Any

from src.application.services.direction_loss_tracker import get_direction_loss_tracker
from src.application.services.execution_market_confluence import _extract_indicator_float
from src.application.services.execution_price_action import _resolve_candle_ohlc
from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


def resolve_closed_candle_direction(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    symbol: str | None = None,
) -> str | None:
    """Extrai a direcao da vela M5 fechada com fallback robusto para OHLC."""
    direct = str(metrics.get("closed_micro_candle_dir") or metrics.get("scale_micro_bar_dir") or "").strip().upper()
    if direct in _VALID:
        return direct
    ohlc = _resolve_candle_ohlc(metrics, orch=orch, symbol=symbol)
    if ohlc is not None:
        open_px, _, _, close_px = ohlc
        if close_px > open_px:
            return TradeDirection.CALL.name
        if close_px < open_px:
            return TradeDirection.PUT.name
    return None


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
    metrics: dict[str, Any] | None = None,
) -> tuple[TradeDirection, str] | None:
    """Detecta rejeicao severa por pavio em extremidade com filtro institucional de tendencia."""
    open_px, high_px, low_px, close_px = ohlc
    range_px = high_px - low_px
    if range_px <= 1e-12:
        return None
    trend = str((metrics or {}).get("trend_direction") or "").strip().upper()
    adx = _extract_indicator_float(metrics or {}, "adx") or _extract_indicator_float(metrics or {}, "adx_norm")
    rsi = _extract_indicator_float(metrics or {}, "rsi")
    if exec_dir == TradeDirection.CALL:
        upper_wick = (high_px - max(open_px, close_px)) / range_px
        if upper_wick < 0.45:
            return None
        if trend == TradeDirection.CALL.name and close_px > open_px:
            if (adx is not None and adx >= 0.25) and (rsi is None or rsi < 0.70):
                return None
            if upper_wick < 0.55:
                return None
        return TradeDirection.PUT, "wick_rejection"
    lower_wick = (min(open_px, close_px) - low_px) / range_px
    if lower_wick < 0.45:
        return None
    if trend == TradeDirection.PUT.name and close_px < open_px:
        if (adx is not None and adx >= 0.25) and (rsi is None or rsi > 0.30):
            return None
        if lower_wick < 0.55:
            return None
    return TradeDirection.CALL, "wick_rejection"


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
    _ohlc: tuple[float, float, float, float],
    metrics: dict[str, Any],
) -> tuple[TradeDirection, str] | None:
    """Detecta exaustao climatica extrema e reverte para mean reversion."""
    rsi = _extract_indicator_float(metrics, "rsi")
    bb_b = _extract_indicator_float(metrics, "bb_pct_b")
    adx = _extract_indicator_float(metrics, "adx") or _extract_indicator_float(metrics, "adx_norm")
    is_strong_trend = adx is not None and adx >= 0.30
    if exec_dir == TradeDirection.CALL:
        if rsi is not None and rsi >= 0.80:
            return TradeDirection.PUT, "climactic_exhaustion"
        if not is_strong_trend and bb_b is not None and bb_b >= 1.10:
            return TradeDirection.PUT, "climactic_exhaustion"
    elif exec_dir == TradeDirection.PUT:
        if rsi is not None and rsi <= 0.20:
            return TradeDirection.CALL, "climactic_exhaustion"
        if not is_strong_trend and bb_b is not None and bb_b <= -0.10:
            return TradeDirection.CALL, "climactic_exhaustion"
    return None


def _check_chop_confluence(
    exec_dir: TradeDirection,
    metrics: dict[str, Any],
) -> tuple[TradeDirection, str] | None:
    """Detecta canal lateral / chop e reverte operacao em suporte/resistencia."""
    adx = _extract_indicator_float(metrics, "adx") or _extract_indicator_float(metrics, "adx_norm")
    is_chop = bool(metrics.get("micro_chop_congestion")) or (adx is not None and adx <= 0.20)
    if not is_chop:
        return None
    rsi = _extract_indicator_float(metrics, "rsi")
    bb_b = _extract_indicator_float(metrics, "bb_pct_b")
    if exec_dir == TradeDirection.PUT and ((rsi is not None and rsi <= 0.45) or (bb_b is not None and bb_b <= 0.40)):
        return TradeDirection.CALL, "chop_support_bounce"
    if exec_dir == TradeDirection.CALL and ((rsi is not None and rsi >= 0.55) or (bb_b is not None and bb_b >= 0.60)):
        return TradeDirection.PUT, "chop_resistance_reversal"
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
    chop = _check_chop_confluence(exec_dir, metrics)
    if chop is not None:
        return chop[0], True, chop[1]
    trend = str(metrics.get("trend_direction") or "").strip().upper()
    candle = resolve_closed_candle_direction(metrics, orch=orch, symbol=symbol)
    ohlc = _resolve_candle_ohlc(metrics, orch=orch, symbol=symbol)
    adx = _extract_indicator_float(metrics, "adx") or _extract_indicator_float(metrics, "adx_norm")
    if trend in _VALID and trend == exec_dir.name and adx is not None and adx >= 0.30:
        if ohlc is not None:
            climax = _check_climactic_confluence(exec_dir, ohlc, metrics)
            if climax is not None:
                return climax[0], True, climax[1]
        return exec_dir, False, None
    if symbol and candle in _VALID and candle != exec_dir.name:
        tracker = get_direction_loss_tracker()
        if tracker.consecutive_losses(str(symbol), exec_dir.name) >= 1:
            return TradeDirection[candle], True, "post_loss_candle_flow"
    if ohlc is not None:
        climax = _check_climactic_confluence(exec_dir, ohlc, metrics)
        if climax is not None:
            return climax[0], True, climax[1]
    if trend in _VALID and trend != exec_dir.name:
        if symbol:
            trend_losses = get_direction_loss_tracker().consecutive_losses(str(symbol), trend)
            if trend_losses >= 1 and candle != trend:
                return exec_dir, False, None
            exec_losses = get_direction_loss_tracker().consecutive_losses(str(symbol), exec_dir.name)
            if exec_losses >= 1:
                return TradeDirection[trend], True, "anti_counter_trend_loss"
        if ohlc is not None:
            climax = _check_climactic_confluence(TradeDirection[trend], ohlc, metrics)
            if climax is not None and climax[0] == exec_dir:
                return climax[0], True, climax[1]
            wick = _check_wick_confluence(TradeDirection[trend], ohlc, metrics)
            if wick is not None and wick[0] == exec_dir:
                return wick[0], True, wick[1]
        if candle == trend:
            return TradeDirection[trend], True, "trend_candle_alignment"
        p_loss = metrics.get("loss_clf_p_loss")
        if p_loss is not None:
            try:
                if float(p_loss) >= 0.50:
                    return TradeDirection[trend], True, "loss_clf_macro_discord"
            except (TypeError, ValueError):
                pass
        return TradeDirection[trend], True, "trend_pullback_resumption"
    if ohlc is not None:
        maru = _check_marubozu_confluence(exec_dir, ohlc)
        if maru is not None:
            return maru[0], True, maru[1]
        wick = _check_wick_confluence(exec_dir, ohlc, metrics)
        if wick is not None:
            return wick[0], True, wick[1]
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
