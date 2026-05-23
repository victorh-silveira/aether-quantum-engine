"""Agregacao diaria e runtime simulado do backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.risk.stop_win_target import resolve_stop_win_target


M15_CANDLE_MINUTES = 15


@dataclass(frozen=True)
class DailySessionStats:
    """Resultado agregado de um dia simulado."""

    day_index: int
    bankroll_start: float
    bankroll_end: float
    pnl: float
    trades: int
    wins: int
    stop_win_target: float
    stop_win_hit: bool
    first_trade_bar: int | None
    stop_win_hit_bar: int | None
    runtime_m15_candles: int | None
    runtime_minutes: float | None
    runtime_label: str


def _format_runtime_minutes(minutes: float) -> str:
    """Formata duracao simulada estilo sessao ao vivo."""
    total = max(0, int(round(minutes)))
    if total < 60:
        return f"{total}m"
    hours = total // 60
    mins = total % 60
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"


def runtime_to_stop_win(
    day_trades: list[Any],
    target: float,
) -> tuple[int | None, int | None, float | None, str]:
    """Calcula velas M15 e minutos ate o stop win (1 vela = 15m)."""
    if not day_trades:
        return None, None, None, "-"
    ordered = sorted(day_trades, key=lambda t: t.bar_index)
    first_bar = ordered[0].bar_index
    stop_bar: int | None = None
    cum = 0.0
    for trade in ordered:
        cum += trade.pnl
        if cum >= target:
            stop_bar = trade.bar_index
            break
    if stop_bar is None:
        return first_bar, None, None, "-"
    candles = stop_bar - first_bar + 1
    minutes = float(candles * M15_CANDLE_MINUTES)
    return first_bar, stop_bar, minutes, _format_runtime_minutes(minutes)


def build_daily_sessions(
    settled: list[Any],
    *,
    bankroll_start: float,
    risk_config: dict[str, Any],
) -> list[DailySessionStats]:
    """Agrega PnL, stop win e runtime simulado por dia."""
    if not settled:
        return []
    by_day: dict[int, list[Any]] = {}
    for trade in settled:
        by_day.setdefault(trade.session_day, []).append(trade)

    sessions: list[DailySessionStats] = []
    running_bankroll = float(bankroll_start)
    for day_index in sorted(by_day.keys()):
        day_trades = by_day[day_index]
        day_start = running_bankroll
        target = resolve_stop_win_target(risk_config, day_start)
        pnl = sum(t.pnl for t in day_trades)
        day_end = day_trades[-1].bankroll_after
        first_bar, stop_bar, runtime_min, runtime_label = runtime_to_stop_win(day_trades, target)
        runtime_candles = None
        if first_bar is not None and stop_bar is not None:
            runtime_candles = stop_bar - first_bar + 1
        sessions.append(
            DailySessionStats(
                day_index=day_index,
                bankroll_start=round(day_start, 2),
                bankroll_end=round(day_end, 2),
                pnl=round(pnl, 2),
                trades=len(day_trades),
                wins=sum(1 for t in day_trades if t.won),
                stop_win_target=round(target, 2),
                stop_win_hit=pnl >= target,
                first_trade_bar=first_bar,
                stop_win_hit_bar=stop_bar,
                runtime_m15_candles=runtime_candles,
                runtime_minutes=runtime_min,
                runtime_label=runtime_label,
            )
        )
        running_bankroll = day_end
    return sessions
