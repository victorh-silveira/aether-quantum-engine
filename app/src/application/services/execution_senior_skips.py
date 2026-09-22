"""Aplicacao fail-closed das salvaguardas de mercado e price action."""

from __future__ import annotations

from typing import Any

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
from src.application.services.execution_signal_skips import should_skip_neg_edge, should_skip_trend_discord
from src.domain.models.trade import TradeDirection


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
        lambda: should_skip_neg_edge(metrics, exec_cfg, force=force),
    )
    return (exec_dir, True) if any(skip() for skip in skips) else (exec_dir, False)
