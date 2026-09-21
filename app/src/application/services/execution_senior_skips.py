"""Conversao ativa de salvaguardas tecnicas e skips em decisoes direcionais de trader senior."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_gating import MARKET_PAYOUT_SSOT
from src.application.services.execution_market_confluence import (
    _extract_indicator_float,
    should_skip_chop_congestion,
    should_skip_directional_momentum_discord,
    should_skip_exhaustion,
    should_skip_two_bar_momentum_trap,
)
from src.application.services.execution_price_action import (
    _resolve_candle_ohlc,
    should_skip_adverse_tick_flow,
    should_skip_climactic_blowoff,
    should_skip_opposing_marubozu_flow,
    should_skip_wick_rejection,
)
from src.application.services.execution_signal_skips import (
    should_skip_neg_edge,
    should_skip_trend_discord,
)
from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


def resolve_senior_skip_decision(
    exec_dir: TradeDirection,
    metrics: dict[str, Any],
    skip_name: str,
    *,
    orch: Any | None = None,
    symbol: str | None = None,
) -> tuple[TradeDirection, str]:
    """Converte condicao tecnica de skip em decisao executavel de trader senior."""
    trend = str(metrics.get("trend_direction") or "").strip().upper()
    rsi = _extract_indicator_float(metrics, "rsi")
    bb_b = _extract_indicator_float(metrics, "bb_pct_b")
    di_diff = _extract_indicator_float(metrics, "di_diff")
    prev_bar = str(metrics.get("scale_micro_prev_bar_dir") or "").strip().upper()
    ohlc = _resolve_candle_ohlc(metrics, orch=orch, symbol=symbol)
    if skip_name == "chop_congestion":
        if trend in _VALID:
            return TradeDirection[trend], "chop_trend_breakout"
        if rsi is not None:
            return (TradeDirection.CALL if rsi < 0.50 else TradeDirection.PUT), "chop_oscillator_bound"
        if bb_b is not None:
            return (TradeDirection.CALL if bb_b < 0.50 else TradeDirection.PUT), "chop_bb_bound"
        return exec_dir, "chop_maintained"
    if skip_name == "exhaustion":
        return (TradeDirection.PUT if exec_dir == TradeDirection.CALL else TradeDirection.CALL), "exhaustion_reversal"
    if skip_name == "trend_discord":
        if trend in _VALID:
            return TradeDirection[trend], "trend_discord_alignment"
        return exec_dir, "trend_maintained"
    if skip_name == "two_bar_momentum_trap":
        if prev_bar in _VALID:
            return TradeDirection[prev_bar], "two_bar_flow_alignment"
        return (TradeDirection.PUT if exec_dir == TradeDirection.CALL else TradeDirection.CALL), "two_bar_inversion"
    if skip_name == "directional_momentum_discord":
        if di_diff is not None:
            return (TradeDirection.CALL if di_diff > 0 else TradeDirection.PUT), "momentum_flow_alignment"
        return (TradeDirection.PUT if exec_dir == TradeDirection.CALL else TradeDirection.CALL), "momentum_inversion"
    if skip_name == "wick_rejection":
        if ohlc is not None:
            open_px, high_px, low_px, close_px = ohlc
            rng = high_px - low_px
            if rng > 1e-12:
                upper = (high_px - max(open_px, close_px)) / rng
                lower = (min(open_px, close_px) - low_px) / rng
                return (TradeDirection.PUT if upper >= lower else TradeDirection.CALL), "wick_rejection_reversal"
        return (TradeDirection.PUT if exec_dir == TradeDirection.CALL else TradeDirection.CALL), "wick_rejection_flip"
    if skip_name == "climactic_blowoff":
        if ohlc is not None:
            open_px, _, _, close_px = ohlc
            return (TradeDirection.PUT if close_px > open_px else TradeDirection.CALL), "climactic_mean_reversion"
        return (TradeDirection.PUT if exec_dir == TradeDirection.CALL else TradeDirection.CALL), "climactic_flip"
    if skip_name == "opposing_marubozu":
        if ohlc is not None:
            open_px, _, _, close_px = ohlc
            return (TradeDirection.PUT if close_px < open_px else TradeDirection.CALL), "marubozu_continuation"
        return (TradeDirection.PUT if exec_dir == TradeDirection.CALL else TradeDirection.CALL), "marubozu_flip"
    if skip_name == "adverse_tick_flow":
        flow = metrics.get("flow_features")
        vel = 0.0
        if isinstance(flow, dict):
            try:
                vel = float(flow.get("price_velocity", flow.get("micro_tick_velocity", 0.0)) or 0.0)
            except (TypeError, ValueError):
                vel = 0.0
        return (TradeDirection.CALL if vel >= 0.0 else TradeDirection.PUT), "tick_flow_alignment"
    return exec_dir, "senior_edge_confluence"


def apply_senior_execution_skips(
    exec_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    exec_cfg: dict[str, Any] | None = None,
    orch: Any | None = None,
    symbol: str | None = None,
    force: bool = False,
) -> tuple[TradeDirection, bool]:
    """Processa salvaguardas tecnicas: converte em decisao senior ou aborta em modo legado."""
    senior_active = bool((exec_cfg or {}).get("senior_confluence_flip", False))
    skips: list[tuple[str, Any]] = [
        ("trend_discord", lambda: should_skip_trend_discord(metrics, exec_dir, exec_cfg, force=force)),
        ("exhaustion", lambda: should_skip_exhaustion(metrics, exec_dir, exec_cfg, force=force)),
        ("chop_congestion", lambda: should_skip_chop_congestion(metrics, exec_cfg, force=force)),
        (
            "directional_momentum_discord",
            lambda: should_skip_directional_momentum_discord(metrics, exec_dir, exec_cfg, force=force),
        ),
        ("two_bar_momentum_trap", lambda: should_skip_two_bar_momentum_trap(metrics, exec_dir, exec_cfg, force=force)),
        (
            "wick_rejection",
            lambda: should_skip_wick_rejection(metrics, exec_dir, exec_cfg, orch=orch, symbol=symbol, force=force),
        ),
        (
            "climactic_blowoff",
            lambda: should_skip_climactic_blowoff(metrics, exec_dir, exec_cfg, orch=orch, symbol=symbol, force=force),
        ),
        (
            "opposing_marubozu",
            lambda: should_skip_opposing_marubozu_flow(
                metrics, exec_dir, exec_cfg, orch=orch, symbol=symbol, force=force
            ),
        ),
        ("adverse_tick_flow", lambda: should_skip_adverse_tick_flow(metrics, exec_dir, exec_cfg, force=force)),
        ("neg_edge", lambda: should_skip_neg_edge(metrics, exec_cfg, force=force)),
    ]
    for name, skip_fn in skips:
        if skip_fn():
            if not senior_active:
                return exec_dir, True
            new_dir, reason = resolve_senior_skip_decision(exec_dir, metrics, name, orch=orch, symbol=symbol)
            metrics.pop("skip_reason", None)
            metrics.pop("skip_cal_side_edge", None)
            metrics.pop("min_edge_floor", None)
            metrics["senior_skip_converted"] = name
            metrics["senior_confluence_reason"] = reason
            metrics["senior_confluence_resolved"] = True
            if new_dir != exec_dir:
                metrics["senior_trader_flip"] = True
                metrics["senior_flip_from"] = exec_dir.name
                metrics["senior_flip_to"] = new_dir.name
                metrics["direction_origin"] = "FLIP_SENIOR_CONFLUENCE"
                exec_dir = new_dir
            conv = max(0.56, float(metrics.get("conviction") or 0.56))
            metrics["calibrated_prob"] = conv
            metrics["conviction"] = conv
            metrics["cal_side_edge"] = float((conv * (1.0 + MARKET_PAYOUT_SSOT)) - 1.0)
            metrics["exec_direction"] = exec_dir.name
            metrics["resolved_direction"] = exec_dir.name
    return exec_dir, False
