"""Simulacao HFT no backtest a partir de payloads Gemini em cache."""

from __future__ import annotations

from typing import Any

from scripts.backtest.hft_walkforward import collect_hft_orders_walkforward


async def collect_hft_with_resolver(
    *,
    m15: dict[str, list[float]],
    m5: dict[str, list[float]],
    us_syms: list[str],
    eu_syms: list[str],
    all_syms: list[str],
    anchor: str,
    macro_cfg: dict[str, Any] | None,
    start: int,
    end: int,
    resolver,
    config: dict[str, Any],
) -> tuple[list, dict[str, Any]]:
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
