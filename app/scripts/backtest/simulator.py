"""Simulacao de PnL RISE_FALL 15m para ordens do backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.backtest.signal_engine import BacktestOrder
from scripts.backtest.simulator_sessions import (
    M15_CANDLE_MINUTES,
    DailySessionStats,
    build_daily_sessions,
    runtime_to_stop_win,
)
from src.domain.models.trade import TradeDirection
from src.domain.risk.risk_manager import RiskManager
from src.domain.risk.stop_win_target import resolve_stop_win_target


M15_BAR_SECONDS = 900
M15_BARS_PER_DAY = 96


@dataclass(frozen=True)
class SettledTrade:
    """Trade com resultado na barra seguinte."""

    bar_index: int
    symbol: str
    direction: TradeDirection
    conviction: float
    macro_tag: str
    active_region: str
    return_pct: float
    won: bool
    pnl: float
    stake: float
    bankroll_after: float
    session_day: int


@dataclass(frozen=True)
class SimulationResult:
    """Resultado da liquidacao com curva de equity."""

    trades: list[SettledTrade]
    equity_curve: list[float]
    max_drawdown_abs: float
    max_drawdown_pct: float
    skipped_stake_zero: int
    skipped_drawdown_brake: int
    skipped_stop_win: int
    daily_sessions: list[DailySessionStats]


def _simulated_session_day(bar_index: int) -> int:
    """Indice de dia simulado (96 velas M15 por dia)."""
    return int(bar_index) // M15_BARS_PER_DAY


def bar_return_pct(closes: list[float], bar_index: int) -> float | None:
    """Retorno percentual close[t+1] vs close[t]."""
    if bar_index + 1 >= len(closes) or bar_index < 0:
        return None
    start = float(closes[bar_index])
    end = float(closes[bar_index + 1])
    if start == 0:
        return None
    return ((end - start) / start) * 100.0


def direction_wins(direction: TradeDirection, return_pct: float) -> bool:
    """True se CALL sobe ou PUT cai no intervalo."""
    if direction == TradeDirection.CALL:
        return return_pct > 0
    return return_pct < 0


def compute_max_drawdown(equity: list[float]) -> tuple[float, float]:
    """Retorna drawdown absoluto e percentual sobre o pico."""
    if not equity:
        return 0.0, 0.0
    peak = equity[0]
    max_dd_abs = 0.0
    for value in equity:
        peak = max(peak, value)
        dd = peak - value
        max_dd_abs = max(max_dd_abs, dd)
    max_dd_pct = (max_dd_abs / peak * 100.0) if peak > 0 else 0.0
    return max_dd_abs, max_dd_pct


def _apply_trade_outcome(rm: RiskManager, symbol: str, profit: float) -> None:
    """Atualiza estado Kelly/recuperacao apos um trade isolado."""
    rm.total_session_profit += profit
    rm.record_trade_outcome(symbol, won=profit >= 0.0)
    if profit < 0:
        rm.pending_loss[symbol] = rm.pending_loss.get(symbol, 0.0) + abs(profit)
        rm.consecutive_losses += 1
    else:
        current_loss = rm.pending_loss.get(symbol, 0.0)
        rm.pending_loss[symbol] = max(0.0, current_loss - profit)
        rm.consecutive_losses = 0


def _sorted_orders(orders: list[BacktestOrder]) -> list[BacktestOrder]:
    """Ordena ordens por barra e simbolo para simulacao sequencial."""
    return sorted(orders, key=lambda o: (o.bar_index, o.symbol))


def settle_orders(
    orders: list[BacktestOrder],
    m15_closes: dict[str, list[float]],
    *,
    stake: float,
    payout: float,
    bankroll_start: float = 100.0,
    risk_config: dict[str, Any] | None = None,
) -> SimulationResult:
    """Liquida ordens com stake fixa respeitando stop win diario."""
    rm = RiskManager(risk_config or {})
    rm.set_initial_bankroll(bankroll_start)
    bankroll = float(bankroll_start)
    equity = [bankroll]
    settled: list[SettledTrade] = []
    skipped = 0
    skipped_sw = 0
    current_day: int | None = None
    cfg = risk_config or {}

    for order in _sorted_orders(orders):
        day = _simulated_session_day(order.bar_index)
        if current_day is None or day != current_day:
            rm.reset_daily_session(bankroll)
            current_day = day

        target = resolve_stop_win_target(cfg, rm.initial_bankroll)
        if rm.total_session_profit >= target:
            skipped += 1
            skipped_sw += 1
            continue

        series = m15_closes.get(order.symbol, [])
        ret = bar_return_pct(series, order.bar_index)
        if ret is None:
            continue
        won = direction_wins(order.direction, ret)
        pnl = stake * payout if won else -stake
        bankroll += pnl
        _apply_trade_outcome(rm, order.symbol, pnl)
        equity.append(bankroll)
        settled.append(
            SettledTrade(
                bar_index=order.bar_index,
                symbol=order.symbol,
                direction=order.direction,
                conviction=order.conviction,
                macro_tag=order.macro_tag,
                active_region=order.active_region,
                return_pct=ret,
                won=won,
                pnl=pnl,
                stake=stake,
                bankroll_after=bankroll,
                session_day=day,
            )
        )
    max_dd_abs, max_dd_pct = compute_max_drawdown(equity)
    daily = build_daily_sessions(settled, bankroll_start=bankroll_start, risk_config=cfg)
    return SimulationResult(
        trades=settled,
        equity_curve=equity,
        max_drawdown_abs=max_dd_abs,
        max_drawdown_pct=max_dd_pct,
        skipped_stake_zero=skipped,
        skipped_drawdown_brake=0,
        skipped_stop_win=skipped_sw,
        daily_sessions=daily,
    )


def settle_orders_kelly(
    orders: list[BacktestOrder],
    m15_closes: dict[str, list[float]],
    risk_config: dict[str, Any],
    *,
    bankroll_start: float,
    payout: float | None = None,
) -> SimulationResult:
    """Liquida ordens com Kelly, stop win e drawdown reiniciados a cada dia."""
    rm = RiskManager(risk_config)
    rm.set_initial_bankroll(bankroll_start)
    if payout is not None:
        rm.risk_params["payout_estimate"] = payout

    b = float(rm.risk_params.get("payout_estimate", 0.95))
    bankroll = float(bankroll_start)
    equity = [bankroll]
    settled: list[SettledTrade] = []
    skipped = 0
    skipped_dd = 0
    skipped_sw = 0
    current_day: int | None = None

    for order in _sorted_orders(orders):
        day = _simulated_session_day(order.bar_index)
        if current_day is None or day != current_day:
            rm.reset_daily_session(bankroll)
            current_day = day

        target = resolve_stop_win_target(risk_config, rm.initial_bankroll)
        if rm.total_session_profit >= target:
            skipped += 1
            skipped_sw += 1
            continue

        series = m15_closes.get(order.symbol, [])
        ret = bar_return_pct(series, order.bar_index)
        if ret is None:
            continue
        stake = rm.calculate_stake(
            bankroll,
            order.symbol,
            order.conviction,
            silent=True,
            apply_stop_win=True,
        )
        if stake <= 0:
            skipped += 1
            if rm.session_max_drawdown_pct > 0 and rm.peak_bankroll > 0:
                dd_pct = ((rm.peak_bankroll - bankroll) / rm.peak_bankroll) * 100.0
                if dd_pct >= rm.session_max_drawdown_pct:
                    skipped_dd += 1
            elif rm.total_session_profit >= target:
                skipped_sw += 1
            continue
        won = direction_wins(order.direction, ret)
        pnl = stake * b if won else -stake
        bankroll += pnl
        _apply_trade_outcome(rm, order.symbol, pnl)
        equity.append(bankroll)
        settled.append(
            SettledTrade(
                bar_index=order.bar_index,
                symbol=order.symbol,
                direction=order.direction,
                conviction=order.conviction,
                macro_tag=order.macro_tag,
                active_region=order.active_region,
                return_pct=ret,
                won=won,
                pnl=pnl,
                stake=stake,
                bankroll_after=bankroll,
                session_day=day,
            )
        )

    max_dd_abs, max_dd_pct = compute_max_drawdown(equity)
    daily = build_daily_sessions(settled, bankroll_start=bankroll_start, risk_config=risk_config)
    return SimulationResult(
        trades=settled,
        equity_curve=equity,
        max_drawdown_abs=max_dd_abs,
        max_drawdown_pct=max_dd_pct,
        skipped_stake_zero=skipped,
        skipped_drawdown_brake=skipped_dd,
        skipped_stop_win=skipped_sw,
        daily_sessions=daily,
    )


__all__ = [
    "DailySessionStats",
    "M15_BAR_SECONDS",
    "M15_BARS_PER_DAY",
    "M15_CANDLE_MINUTES",
    "SettledTrade",
    "SimulationResult",
    "bar_return_pct",
    "compute_max_drawdown",
    "direction_wins",
    "runtime_to_stop_win",
    "settle_orders",
    "settle_orders_kelly",
]
