"""Simulacao HFT no backtest a partir de payloads Gemini em cache."""

from __future__ import annotations

from typing import Any

from scripts.backtest.hft_cycle import cooldown_slots, hft_slots_per_m15_bar
from scripts.backtest.signal_engine import BacktestOrder
from scripts.backtest.snapshot_engine import build_snapshot_at_bar


async def collect_hft_with_resolver(
    *,
    m15: dict[str, list[float]],
    m5: dict[str, list[float]],
    us_syms: list[str],
    eu_syms: list[str],
    macro_cfg: dict[str, Any] | None,
    start: int,
    end: int,
    resolver,
    config: dict[str, Any],
) -> tuple[list[BacktestOrder], dict[str, Any]]:
    """Percorre barras M15 e aplica resolver para gerar ordens HFT."""
    slots = hft_slots_per_m15_bar(config)
    cool = cooldown_slots(config, slots_per_bar=slots)
    orders: list[BacktestOrder] = []
    bars_with_signal = 0
    open_until_bar: int | None = None
    last_entry_slot = -10_000

    for bar_index in range(start, end + 1):
        snap = build_snapshot_at_bar(
            bar_index=bar_index,
            m15_closes=m15,
            m5_closes=m5,
            us_symbols=us_syms,
            eu_symbols=eu_syms,
            macro_cfg=macro_cfg,
        )
        bar_orders = await resolver(bar_index, snap)
        if not bar_orders:
            continue
        bars_with_signal += 1
        base_slot = bar_index * slots
        for slot in range(slots):
            global_slot = base_slot + slot
            if open_until_bar is not None and bar_index < open_until_bar:
                break
            if cool > 0 and global_slot - last_entry_slot < cool:
                continue
            orders.append(bar_orders[0])
            open_until_bar = bar_index + 1
            last_entry_slot = global_slot
            break

    stats = {
        "hft_slots_per_m15_bar": slots,
        "hft_cooldown_slots": cool,
        "hft_contract_lock_bars": 1,
        "bars_with_signal": bars_with_signal,
        "signals_generated": len(orders),
    }
    return orders, stats
