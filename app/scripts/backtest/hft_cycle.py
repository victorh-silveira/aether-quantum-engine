"""Cadencia HFT do orchestrator no backtest M15 (walk-forward alinhado ao live)."""

from __future__ import annotations

from typing import Any

from scripts.backtest.backtest_cluster_runtime import BacktestClusterRuntime
from scripts.backtest.hft_walkforward import collect_hft_orders_walkforward
from scripts.backtest.signal_engine import resolve_orders_at_bar


async def collect_hft_orders(
    *,
    config: dict[str, Any],
    m15: dict[str, list[float]],
    m5: dict[str, list[float]],
    us_syms: list[str],
    eu_syms: list[str],
    all_syms: list[str],
    anchor: str,
    macro_cfg: dict[str, Any] | None,
    start: int,
    end: int,
) -> tuple[list[Any], dict[str, Any]]:
    def resolver(bar_index: int, snap: Any, runtime: BacktestClusterRuntime) -> list:
        return resolve_orders_at_bar(
            bar_index=bar_index,
            snapshot=snap,
            config=config,
            us_symbols=us_syms,
            eu_symbols=eu_syms,
            all_symbols=all_syms,
            anchor=anchor,
            runtime=runtime,
        )

    risk_cfg = config.get("risk_management") if isinstance(config.get("risk_management"), dict) else None
    return await collect_hft_orders_walkforward(
        config=config,
        m15=m15,
        m5=m5,
        us_syms=us_syms,
        eu_syms=eu_syms,
        all_syms=all_syms,
        anchor=anchor,
        macro_cfg=macro_cfg,
        start=start,
        end=end,
        resolver=resolver,
        risk_config=risk_cfg,
    )
