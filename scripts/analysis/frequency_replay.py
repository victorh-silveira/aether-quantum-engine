"""Replay causal de portfolio por velas, sem qualificar contratos de corretora."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import groupby

from src.application.services.indicator_consensus import available_cluster_bankroll
from src.domain.fundamental.economic_calendar import CalendarPolicy, EconomicEvent, evaluate_calendar_veto
from src.domain.risk.pullback_stake import pullback_stake


BRT = timezone(timedelta(hours=-3))


@dataclass(frozen=True)
class Signal:
    """Sinal fechado cujo desfecho futuro e usado somente apos a selecao."""

    symbol: str
    entry_epoch: int
    side: str
    outcome: int
    selection_score: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Rejeita dados ambiguos antes do replay."""
        if not self.symbol or self.side not in ("CALL", "PUT") or self.outcome not in (-1, 0, 1):
            raise ValueError("sinal invalido")
        if self.entry_epoch < 0 or self.entry_epoch % 900:
            raise ValueError("entrada deve estar alinhada a M15")
        if any(not math.isfinite(value) for value in self.selection_score):
            raise ValueError("score deve ser finito")


def _day(epoch: int) -> str:
    """Converte epoch para a data operacional BRT."""
    return datetime.fromtimestamp(epoch, tz=BRT).date().isoformat()


def _daily_range(start: int, end: int) -> dict[str, dict]:
    """Inclui dias sem operacoes na janela inclusiva de entradas."""
    daily = {}
    current = datetime.fromtimestamp(start, tz=BRT).date()
    last = datetime.fromtimestamp(end, tz=BRT).date()
    while current <= last:
        daily[current.isoformat()] = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "net_units": 0.0,
            "pnl": 0.0,
        }
        current += timedelta(days=1)
    return daily


def _rank_key(signal: Signal) -> tuple:
    """Usa score observado ou proxy fixo, com empate lexical sem consultar desfecho."""
    return (signal.selection_score, signal.symbol == "OTC_FTSE")


def replay_portfolio(
    signals: list[Signal],
    *,
    payout: float,
    initial_bankroll: float = 1000.0,
    events: tuple[EconomicEvent, ...] = (),
    calendar_policy: CalendarPolicy | None = None,
    calendar_coverage_complete: bool = False,
    start_epoch: int | None = None,
    end_epoch: int | None = None,
) -> dict:
    """Simula compras sequenciais e liquidacao conjunta M15 com sizing de producao.

    Calendario ausente nao cria vetos artificiais. Sem snapshots de corretora,
    o ranking FTSE/lexical e um proxy. Elegibilidade DEMO/REAL, falhas de compra,
    latencia e atraso de settlement nao podem ser reconstruidos com OHLC.
    """
    if not math.isfinite(payout) or payout <= 0 or not math.isfinite(initial_bankroll) or initial_bankroll <= 0:
        raise ValueError("payout e banca devem ser positivos e finitos")
    ordered = sorted(signals, key=lambda signal: signal.entry_epoch)
    keys = [(signal.entry_epoch, signal.symbol) for signal in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("sinal duplicado por ativo/ciclo")
    start = start_epoch if start_epoch is not None else (ordered[0].entry_epoch if ordered else 0)
    end = end_epoch if end_epoch is not None else (ordered[-1].entry_epoch if ordered else start)
    if start > end:
        raise ValueError("janela invalida")
    ordered = [signal for signal in ordered if start <= signal.entry_epoch <= end]
    daily = _daily_range(start, end)
    policy = calendar_policy if calendar_policy is not None else CalendarPolicy()
    counts: defaultdict[str, int] = defaultdict(
        int,
        dict.fromkeys(
            ("raw_signals", "calendar_vetoes", "capacity_rejections", "stake_rejections", "cluster_budget_breaches"), 0
        ),
    )
    trades, cycles = [], []
    balance, peak, drawdown = initial_bankroll, initial_bankroll, 0.0
    consecutive_losses = max_losses = 0
    for epoch, grouped in groupby(ordered, key=lambda signal: signal.entry_epoch):
        group = list(grouped)
        counts["raw_signals"] += len(group)
        eligible = []
        for signal in group:
            veto, _ = evaluate_calendar_veto(signal.symbol, epoch, 900, events, policy)
            if veto:
                counts["calendar_vetoes"] += 1
            else:
                eligible.append(signal)
        ranked = sorted(sorted(eligible, key=lambda signal: signal.symbol), key=_rank_key, reverse=True)
        selected = ranked[:4]
        counts["capacity_rejections"] += max(0, len(ranked) - len(selected))
        snapshot, reserved = balance, 0.0
        cycle_trades = []
        for signal in selected:
            available = available_cluster_bankroll(snapshot, reserved, snapshot - reserved, multi_symbol=True)
            stake, _ = pullback_stake(
                available,
                0.0,
                payout,
                1.0,
                amort_steps=0,
                base_stake_pct=0.015,
                cycle_risk_budget_pct=0.04,
                cycle_order_count=len(selected),
            )
            if stake <= 0 or stake > available:
                counts["stake_rejections"] += 1
                continue
            reserved += stake
            net_units = payout if signal.outcome == 1 else float(signal.outcome)
            pnl = round(stake * net_units, 2)
            cycle_trades.append(
                {
                    "symbol": signal.symbol,
                    "entry_epoch": epoch,
                    "settlement_epoch": epoch + 900,
                    "side": signal.side,
                    "outcome": signal.outcome,
                    "stake": stake,
                    "net_units": net_units,
                    "pnl": pnl,
                }
            )
        if reserved > snapshot * 0.04 + 1e-9:
            counts["cluster_budget_breaches"] += 1
        cycle_pnl = round(sum(trade["pnl"] for trade in cycle_trades), 2)
        balance = round(snapshot + cycle_pnl, 2)
        peak = max(peak, balance)
        drawdown = max(drawdown, (peak - balance) / peak)
        stats = daily[_day(epoch)]
        for trade in cycle_trades:
            outcome = trade["outcome"]
            stats["trades"] += 1
            stats[{1: "wins", -1: "losses", 0: "ties"}[outcome]] += 1
            stats["net_units"] += trade["net_units"]
            stats["pnl"] += trade["pnl"]
            consecutive_losses = consecutive_losses + 1 if outcome == -1 else 0
            max_losses = max(max_losses, consecutive_losses)
        trades.extend(cycle_trades)
        cycles.append(
            {
                "entry_epoch": epoch,
                "signals": len(group),
                "calendar_eligible": len(eligible),
                "selected": len(selected),
                "executed": len(cycle_trades),
                "snapshot": snapshot,
                "reserved_stake": round(reserved, 2),
                "pnl": cycle_pnl,
                "balance": balance,
            }
        )
    count = len(trades)
    net = sum(trade["net_units"] for trade in trades)
    return {
        "qualifies_contract": False,
        "calendar_coverage_complete": calendar_coverage_complete,
        "ranking": "provided_score"
        if ordered and all(signal.selection_score for signal in ordered)
        else "proxy_FTSE_lexical",
        "limitations": [
            "Proxy OHLC sem cotacoes, elegibilidade broker, latencia, falhas de compra ou settlement atrasado.",
            "Desempates lexicais; producao depende tambem da conclusao assincrona de propostas.",
            "Drawdown sobre patrimonio liquidado, sem marcacao a mercado intravela.",
            "Sequencia de perdas simultaneas depende do desempate deterministico adotado.",
            "Piso broker de 1 unidade pode exceder budget percentual em bancas pequenas; ocorrencias sao contadas.",
        ],
        "payout": payout,
        "initial_bankroll": initial_bankroll,
        "final_bankroll": balance,
        "return_pct": (balance / initial_bankroll - 1) * 100,
        "max_drawdown_pct": drawdown * 100,
        "max_consecutive_losses": max_losses,
        "trade_count": count,
        "net_units": net,
        "ev": net / count if count else 0.0,
        "mean_daily_trades": count / len(daily),
        "zero_trade_days": sum(row["trades"] == 0 for row in daily.values()),
        "active_days": sum(row["trades"] > 0 for row in daily.values()),
        "counts": dict(counts),
        "daily": daily,
        "trades": trades,
        "cycles": cycles,
    }
