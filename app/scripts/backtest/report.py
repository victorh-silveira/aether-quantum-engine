"""Relatorio agregado do backtest Medallion."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.backtest.simulator import SettledTrade, SimulationResult


def _profit_factor(trades: list[SettledTrade]) -> float:
    """Calcula profit factor (ganhos / perdas absolutas)."""
    wins = sum(t.pnl for t in trades if t.pnl > 0)
    losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if losses <= 0:
        return float(wins) if wins > 0 else 0.0
    return wins / losses


def build_report(
    sim: SimulationResult,
    *,
    meta: dict[str, Any],
    bankroll_start: float,
    sizing_mode: str,
) -> dict[str, Any]:
    """Monta estrutura JSON do relatorio."""
    settled = sim.trades
    total = len(settled)
    wins = sum(1 for t in settled if t.won)
    total_pnl = sum(t.pnl for t in settled)
    bankroll_end = settled[-1].bankroll_after if settled else bankroll_start + total_pnl
    avg_stake = (sum(t.stake for t in settled) / total) if total else 0.0

    by_tag: dict[str, list[SettledTrade]] = defaultdict(list)
    by_symbol: dict[str, list[SettledTrade]] = defaultdict(list)
    for t in settled:
        by_tag[t.macro_tag].append(t)
        by_symbol[t.symbol].append(t)

    def bucket_stats(items: list[SettledTrade]) -> dict[str, Any]:
        n = len(items)
        w = sum(1 for x in items if x.won)
        pnl = sum(x.pnl for x in items)
        return {
            "trades": n,
            "wins": w,
            "win_rate": (w / n) if n else 0.0,
            "pnl": pnl,
            "profit_factor": _profit_factor(items),
        }

    daily_rows = [
        {
            "day": d.day_index + 1,
            "bankroll_start": d.bankroll_start,
            "bankroll_end": d.bankroll_end,
            "pnl": d.pnl,
            "trades": d.trades,
            "wins": d.wins,
            "win_rate": (d.wins / d.trades) if d.trades else 0.0,
            "stop_win_target": d.stop_win_target,
            "stop_win_hit": d.stop_win_hit,
            "first_trade_bar": d.first_trade_bar,
            "stop_win_hit_bar": d.stop_win_hit_bar,
            "runtime_m15_candles": d.runtime_m15_candles,
            "runtime_minutes": d.runtime_minutes,
            "runtime_label": d.runtime_label,
        }
        for d in sim.daily_sessions
    ]
    stop_win_days = sum(1 for d in sim.daily_sessions if d.stop_win_hit)
    runtimes = [d.runtime_minutes for d in sim.daily_sessions if d.runtime_minutes is not None]
    median_runtime = sorted(runtimes)[len(runtimes) // 2] if runtimes else None
    stop_win_rate = (stop_win_days / len(sim.daily_sessions)) if sim.daily_sessions else 0.0

    return {
        "meta": meta,
        "summary": {
            "trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total) if total else 0.0,
            "total_pnl": total_pnl,
            "profit_factor": _profit_factor(settled),
            "sizing_mode": sizing_mode,
            "avg_stake": round(avg_stake, 2),
            "bankroll_start": bankroll_start,
            "bankroll_end": bankroll_end,
            "roi_pct": (total_pnl / bankroll_start * 100.0) if bankroll_start > 0 else 0.0,
            "max_drawdown_abs": round(sim.max_drawdown_abs, 2),
            "max_drawdown_pct": round(sim.max_drawdown_pct, 2),
            "skipped_stake_zero": sim.skipped_stake_zero,
            "skipped_drawdown_brake": sim.skipped_drawdown_brake,
            "skipped_stop_win": sim.skipped_stop_win,
            "stop_win_days_hit": stop_win_days,
            "stop_win_hit_rate": round(stop_win_rate, 4),
            "median_runtime_minutes_to_stop_win": median_runtime,
            "simulated_days": len(sim.daily_sessions),
            "signals_generated": meta.get("signals_generated"),
            "bars_with_signal": meta.get("bars_with_signal"),
            "hft_slots_per_m15_bar": meta.get("hft_slots_per_m15_bar"),
        },
        "equity_curve": [round(v, 2) for v in sim.equity_curve],
        "by_day": daily_rows,
        "by_macro_tag": {k: bucket_stats(v) for k, v in sorted(by_tag.items())},
        "by_symbol": {k: bucket_stats(v) for k, v in sorted(by_symbol.items())},
        "trades": [
            {
                **{k: v for k, v in asdict(t).items() if k != "direction"},
                "direction": t.direction.name,
            }
            for t in settled
        ],
    }


def print_summary(report: dict[str, Any]) -> None:
    """Imprime resumo legivel no terminal."""
    s = report["summary"]
    meta = report.get("meta", {})
    mode = str(meta.get("mode", "quant_surrogate"))
    mode_label = "Gemini (prompt live)" if mode == "gemini" else "surrogate quant"
    print(f"=== Backtest Medallion M15 ({mode_label}) ===")
    print(f"Trades: {s['trades']} | Wins: {s['wins']} | Win rate: {s['win_rate']:.1%}")
    print(f"PnL total: {s['total_pnl']:.2f} | Profit factor: {s['profit_factor']:.2f} | ROI: {s['roi_pct']:.2f}%")
    print(f"Banca: ${s['bankroll_start']:.2f} -> ${s['bankroll_end']:.2f} | Sizing: {s['sizing_mode']}")
    if s.get("avg_stake"):
        print(f"Stake media: ${s['avg_stake']:.2f}")
    print(
        f"Max drawdown: ${s['max_drawdown_abs']:.2f} ({s['max_drawdown_pct']:.2f}%)"
        f" | Stake zero: {s.get('skipped_stake_zero', 0)}"
        f" (stop win: {s.get('skipped_stop_win', 0)})"
    )
    if s.get("simulated_days"):
        med = s.get("median_runtime_minutes_to_stop_win")
        med_txt = f"{med:.0f} min" if med is not None else "-"
        print(
            f"Stop win diario: {s.get('stop_win_days_hit', 0)}/{s.get('simulated_days', 0)} dias"
            f" ({s.get('stop_win_hit_rate', 0):.1%}) | mediana ate meta: {med_txt}"
        )
    if mode == "gemini":
        print(
            f"Gemini: chamadas={meta.get('gemini_llm_calls', 0)} "
            f"pontos_agenda={meta.get('gemini_query_points', 0)} "
            f"agenda={meta.get('gemini_schedule', 'daily')} "
            f"falhas={meta.get('gemini_llm_failures', 0)} "
            f"cache={meta.get('gemini_cache_path') or '-'}"
        )
    if meta.get("window_days_requested") is not None or meta.get("m15_bars_aligned"):
        print(
            f"Janela: --days={meta.get('window_days_requested')} "
            f"| M15 alinhadas={meta.get('m15_bars_aligned')} "
            f"| avaliadas={meta.get('bars_evaluated')}"
        )
    if s.get("signals_generated") is not None:
        print(
            f"HFT: {s.get('hft_slots_per_m15_bar', 0)} ciclos/vela M15"
            f" | Sinais: {s.get('signals_generated')} | Barras com sinal: {s.get('bars_with_signal')}"
        )
    by_day = report.get("by_day", [])
    if by_day:
        print("--- Por dia (sessao UTC / 96 velas M15) ---")
        for row in by_day:
            hit = "SIM" if row.get("stop_win_hit") else "nao"
            runtime = row.get("runtime_label") or "-"
            candles = row.get("runtime_m15_candles")
            candle_txt = f"{candles} velas M15" if candles else "sem stop win no dia"
            print(
                f"  Dia {row['day']}: banca ${row['bankroll_start']:.2f}->${row['bankroll_end']:.2f}"
                f" | pnl={row['pnl']:.2f} | trades={row['trades']} wr={row['win_rate']:.1%}"
                f" | meta stop win ${row['stop_win_target']:.2f} ({hit})"
                f" | runtime simulado: {runtime} ({candle_txt})"
            )
    print("--- Por macro tag ---")
    for tag, stats in report.get("by_macro_tag", {}).items():
        print(f"  {tag}: trades={stats['trades']} wr={stats['win_rate']:.1%} pnl={stats['pnl']:.2f}")
    print("--- Por simbolo ---")
    for sym, stats in report.get("by_symbol", {}).items():
        print(f"  {sym}: trades={stats['trades']} wr={stats['win_rate']:.1%} pnl={stats['pnl']:.2f}")


def save_report(path: Path, report: dict[str, Any]) -> None:
    """Grava relatorio JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
