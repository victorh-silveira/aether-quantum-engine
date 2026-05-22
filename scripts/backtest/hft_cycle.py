"""Cadencia HFT do orchestrator (ciclo a cada N segundos) no backtest M15."""

from __future__ import annotations

from typing import Any

from scripts.backtest.signal_engine import BacktestOrder, resolve_orders_at_bar
from scripts.backtest.snapshot_engine import build_snapshot_at_bar


M15_BAR_SECONDS = 900


def hft_slots_per_m15_bar(config: dict[str, Any]) -> int:
    """Quantidade de ciclos de decisao por vela M15 (ex.: 900s / 15s = 60)."""
    orch = config.get("orchestrator", {}) if isinstance(config.get("orchestrator"), dict) else {}
    cycle_iv = max(1, int(orch.get("cycle_interval_seconds", 15)))
    return max(1, M15_BAR_SECONDS // cycle_iv)


def cooldown_slots(config: dict[str, Any], *, slots_per_bar: int) -> int:
    """Converte entry_cooldown_ticks em slots HFT dentro da mesma barra M15."""
    risk = config.get("risk_management", {}) if isinstance(config.get("risk_management"), dict) else {}
    params = risk.get("params", {}) if isinstance(risk.get("params"), dict) else {}
    ticks = max(0, int(params.get("entry_cooldown_ticks", 0)))
    if ticks <= 0:
        return 0
    return min(slots_per_bar, ticks)


def collect_hft_orders(
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
) -> tuple[list[BacktestOrder], dict[str, Any]]:
    """Varre cada slot HFT por vela M15; no maximo 1 entrada por vela (contrato 15m)."""
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
        bar_orders = resolve_orders_at_bar(
            bar_index=bar_index,
            snapshot=snap,
            config=config,
            us_symbols=us_syms,
            eu_symbols=eu_syms,
            all_symbols=all_syms,
            anchor=anchor,
        )
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
