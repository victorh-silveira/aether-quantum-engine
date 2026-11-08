"""Aplicacao fail-closed das salvaguardas de mercado e price action."""

from __future__ import annotations

import contextlib
from typing import Any

from src.application.services.execution_four_vetoes import apply_four_market_vetoes, resolve_four_vetoes_enabled
from src.application.services.execution_market_confluence import (
    should_skip_chop_congestion,
    should_skip_directional_momentum_discord,
    should_skip_exhaustion,
    should_skip_two_bar_momentum_trap,
)
from src.application.services.execution_price_action import (
    should_skip_adverse_tick_flow,
    should_skip_climactic_blowoff,
    should_skip_opposing_marubozu_flow,
    should_skip_wick_rejection,
)
from src.application.services.execution_signal_skips import (
    _closed_candle_dir,
    _mark_skip,
    _pend_waives,
    should_skip_neg_edge,
    should_skip_trend_discord,
)
from src.domain.models.trade import TradeDirection


def should_skip_tcn_noise_discord(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> bool:
    """Bloqueia ordens em zona de puro ruido do modelo contra a vela fechada."""
    if force or not bool((exec_cfg or {}).get("skip_tcn_noise_discord", True)):
        return False
    if bool(metrics.get("loss_clf_flip")) or bool(metrics.get("anti_trend_lock_flip")):
        return False
    margin = metrics.get("direction_margin")
    if margin is None:
        raw_cal = metrics.get("calibrated_prob")
        if raw_cal is not None:
            with contextlib.suppress(TypeError, ValueError):
                margin = abs(float(raw_cal) - 0.5)
    if margin is None:
        return False
    m_val = None
    with contextlib.suppress(TypeError, ValueError):
        m_val = float(margin)
    if m_val is None:
        return False
    floor = float((exec_cfg or {}).get("tcn_noise_margin_floor", 0.012))
    if m_val >= floor:
        return False
    candle = metrics.get("closed_micro_candle_dir")
    if not candle or str(candle).upper() not in {"CALL", "PUT"}:
        return False
    if str(candle).upper() == exec_dir.name:
        return False
    edge = metrics.get("cal_side_edge")
    if edge is not None:
        try:
            if float(edge) > 0.02:
                return False
        except (TypeError, ValueError):
            pass
    _mark_skip(
        metrics,
        "tcn_noise_discord",
        noise_margin=float(m_val),
        candle_dir=str(candle).upper(),
        exec_dir=exec_dir.name,
    )
    return True


def should_skip_explosion_discord(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    exec_cfg: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> bool:
    """Bloqueia ordens contrarias a vela de rompimento em regime de explosao de volatilidade."""
    if force or bool(metrics.get("loss_clf_flip")) or bool(metrics.get("anti_trend_lock_flip")):
        return False
    if exec_cfg is not None and not bool(exec_cfg.get("skip_explosion_discord", True)):
        return False
    if _pend_waives(metrics, exec_cfg):
        return False
    regime = str(metrics.get("scale_micro_regime") or "").strip().lower()
    if regime != "explosion":
        return False
    candle = _closed_candle_dir(metrics)
    if not candle or candle == exec_dir.name:
        return False
    edge = 0.0
    try:
        edge = float(metrics.get("cal_side_edge", metrics.get("edge", 0.0)) or 0.0)
    except (TypeError, ValueError):
        edge = 0.0
    floor = float((exec_cfg or {}).get("explosion_discord_min_edge", 0.08))
    if edge >= floor:
        return False
    _mark_skip(metrics, "explosion_discord", exec_dir=exec_dir.name, candle_dir=candle, edge=float(edge))
    return True


def apply_senior_execution_skips(
    exec_dir: TradeDirection,
    metrics: dict[str, Any],
    *,
    exec_cfg: dict[str, Any] | None = None,
    orch: Any | None = None,
    symbol: str | None = None,
    force: bool = False,
) -> tuple[TradeDirection, bool]:
    """Bloqueia o ciclo quando uma salvaguarda ativa encontrar risco de sinal."""
    if resolve_four_vetoes_enabled(exec_cfg):
        blocked = apply_four_market_vetoes(exec_dir, metrics, orch=orch, symbol=symbol)
        if blocked:
            return exec_dir, True
        if should_skip_tcn_noise_discord(metrics, exec_dir, exec_cfg, force=force):
            return exec_dir, True
        if should_skip_explosion_discord(metrics, exec_dir, exec_cfg, force=force):
            return exec_dir, True
        if should_skip_trend_discord(metrics, exec_dir, exec_cfg, force=force):
            candle = _closed_candle_dir(metrics)
            if candle and candle != exec_dir.name:
                return exec_dir, True
        return exec_dir, should_skip_neg_edge(metrics, exec_cfg, force=force)
    skips = (
        lambda: should_skip_trend_discord(metrics, exec_dir, exec_cfg, force=force),
        lambda: should_skip_exhaustion(metrics, exec_dir, exec_cfg, force=force),
        lambda: should_skip_chop_congestion(metrics, exec_cfg, force=force),
        lambda: should_skip_directional_momentum_discord(metrics, exec_dir, exec_cfg, force=force),
        lambda: should_skip_two_bar_momentum_trap(metrics, exec_dir, exec_cfg, force=force),
        lambda: should_skip_wick_rejection(metrics, exec_dir, exec_cfg, orch=orch, symbol=symbol, force=force),
        lambda: should_skip_climactic_blowoff(metrics, exec_dir, exec_cfg, orch=orch, symbol=symbol, force=force),
        lambda: should_skip_opposing_marubozu_flow(metrics, exec_dir, exec_cfg, orch=orch, symbol=symbol, force=force),
        lambda: should_skip_adverse_tick_flow(metrics, exec_dir, exec_cfg, force=force),
        lambda: should_skip_tcn_noise_discord(metrics, exec_dir, exec_cfg, force=force),
        lambda: should_skip_neg_edge(metrics, exec_cfg, force=force),
    )
    return (exec_dir, True) if any(skip() for skip in skips) else (exec_dir, False)
