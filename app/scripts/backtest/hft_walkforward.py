"""Coleta HFT walk-forward com liquidacao inline (pausa cluster e cooldown como no live)."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from scripts.backtest.backtest_cluster_runtime import BacktestClusterRuntime
from scripts.backtest.hft_slots import cooldown_slots, hft_slots_per_m15_bar
from scripts.backtest.signal_engine import BacktestOrder
from scripts.backtest.simulator import bar_return_pct, direction_wins
from scripts.backtest.snapshot_engine import build_snapshot_at_bar
from scripts.backtest.timeframe import bar_minutes, bars_per_day, primary_granularity_seconds
from src.application.services.orchestrator.trading_session import trading_session_allows_entry
from src.domain.risk.risk_manager import RiskManager


BarResolver = Callable[..., list[BacktestOrder] | Awaitable[list[BacktestOrder]]]


@dataclass
class _WalkforwardCounters:
    bars_with_signal: int = 0
    bars_blocked_session: int = 0
    bars_blocked_pause: int = 0
    cool: int = 0


def _bar_epoch_utc(bar_index: int, config: dict[str, Any]) -> int:
    gran = primary_granularity_seconds(config)
    per_day = bars_per_day(gran)
    minutes = bar_minutes(gran)
    slot = int(bar_index) % per_day
    day = int(bar_index) // per_day
    hour = 7 + (slot * minutes) // 60
    minute = (slot * minutes) % 60
    return day * 86400 + hour * 3600 + minute * 60


async def _resolve_bar_orders(
    resolver: BarResolver,
    bar_index: int,
    snap: Any,
    runtime: BacktestClusterRuntime,
) -> list[BacktestOrder]:
    out = resolver(bar_index, snap, runtime)
    if inspect.isawaitable(out):
        out = await out
    return list(out or [])


def _settle_inline_order(
    *,
    candidate: BacktestOrder,
    bar_index: int,
    global_slot: int,
    m15: dict[str, list[float]],
    runtime: BacktestClusterRuntime,
    rm: RiskManager | None,
) -> None:
    series = m15.get(candidate.symbol, [])
    ret = bar_return_pct(series, bar_index)
    if ret is None:
        return
    won = direction_wins(candidate.direction, ret)
    profit = 1.0 if won else -1.0
    if rm is not None:
        rm.register_result(profit, contract_id=bar_index, symbol=candidate.symbol, current_tick=global_slot)
    if not won:
        runtime.on_trade_loss(symbol=candidate.symbol, direction=candidate.direction)


def _try_hft_entry(
    *,
    bar_index: int,
    slots: int,
    cool: int,
    candidate: BacktestOrder,
    m15: dict[str, list[float]],
    runtime: BacktestClusterRuntime,
    rm: RiskManager | None,
    open_until_bar: int | None,
    last_entry_slot: int,
    counters: _WalkforwardCounters,
) -> tuple[list[BacktestOrder], int | None, int, bool]:
    orders: list[BacktestOrder] = []
    base_slot = bar_index * slots
    for slot in range(slots):
        global_slot = base_slot + slot
        if open_until_bar is not None and bar_index < open_until_bar:
            break
        if cool > 0 and global_slot - last_entry_slot < cool:
            continue
        if runtime._cluster_pause_after_loss_active is True:
            counters.bars_blocked_pause += 1
            break
        if rm is not None and rm.is_on_cooldown(global_slot):
            continue
        orders.append(candidate)
        _settle_inline_order(
            candidate=candidate,
            bar_index=bar_index,
            global_slot=global_slot,
            m15=m15,
            runtime=runtime,
            rm=rm,
        )
        return orders, bar_index + 1, global_slot, True
    return orders, open_until_bar, last_entry_slot, False


async def collect_hft_orders_walkforward(
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
    resolver: BarResolver,
    risk_config: dict[str, Any] | None = None,
) -> tuple[list[BacktestOrder], dict[str, Any]]:
    slots = hft_slots_per_m15_bar(config)
    runtime = BacktestClusterRuntime(config, symbols=all_syms, anchor=anchor)
    rm = RiskManager(risk_config or {}) if risk_config else None
    orders: list[BacktestOrder] = []
    counters = _WalkforwardCounters()
    open_until_bar: int | None = None
    last_entry_slot = -10_000

    for bar_index in range(start, end + 1):
        runtime.begin_cycle()
        epoch = _bar_epoch_utc(bar_index, config)
        allowed_sess, _ = trading_session_allows_entry(
            epoch_utc=epoch,
            stream_ready_at=None,
            now_mono=0.0,
            config=config,
        )
        if not allowed_sess:
            counters.bars_blocked_session += 1
            runtime.end_cycle()
            continue

        snap = build_snapshot_at_bar(
            bar_index=bar_index,
            m15_closes=m15,
            m5_closes=m5,
            us_symbols=us_syms,
            eu_symbols=eu_syms,
            macro_cfg=macro_cfg,
            anchor=anchor,
        )
        bar_orders = await _resolve_bar_orders(resolver, bar_index, snap, runtime)
        if not bar_orders:
            runtime.end_cycle()
            continue
        counters.bars_with_signal += 1
        counters.cool = cooldown_slots(config, slots_per_bar=slots, conviction=float(bar_orders[0].conviction))
        new_orders, open_until_bar, last_entry_slot, _ = _try_hft_entry(
            bar_index=bar_index,
            slots=slots,
            cool=counters.cool,
            candidate=bar_orders[0],
            m15=m15,
            runtime=runtime,
            rm=rm,
            open_until_bar=open_until_bar,
            last_entry_slot=last_entry_slot,
            counters=counters,
        )
        orders.extend(new_orders)
        runtime.end_cycle()

    stats = {
        "hft_slots_per_m15_bar": slots,
        "hft_cooldown_slots": counters.cool,
        "hft_contract_lock_bars": 1,
        "bars_with_signal": counters.bars_with_signal,
        "signals_generated": len(orders),
        "bars_blocked_session": counters.bars_blocked_session,
        "bars_blocked_cluster_pause": counters.bars_blocked_pause,
        "architecture": "live_cluster_walkforward",
    }
    return orders, stats
