"""Formatacao OpenMetrics/Prometheus para metricas de negocio do motor Laya."""

from __future__ import annotations

from typing import Any


def format_business_prometheus_metrics(inst: Any) -> str:
    """Serializa todas as metricas de negocio no formato OpenMetrics com labels semanticos."""
    target_bal = inst._session_start_balance + inst._session_target_win
    profit = (
        (inst._current_balance - inst._session_start_balance) if inst._session_start_balance > 0.0 else inst._total_pnl
    )
    rem = max(0.0, inst._session_target_win - profit)
    prog = min(100.0, max(0.0, (profit / inst._session_target_win * 100.0))) if inst._session_target_win > 0.0 else 0.0
    roi = (profit / inst._session_start_balance * 100.0) if inst._session_start_balance > 0.0 else 0.0
    lines: list[str] = [
        f"aether_trading_balance_usd {round(inst._current_balance, 2)}",
        f"aether_trading_pnl_usd {round(inst._total_pnl, 4)}",
        f"aether_trading_max_drawdown_pct {round(inst._max_drawdown_pct, 2)}",
        f"aether_indicator_confirmed_setups_count {inst._confirmed_setups_count}",
    ]
    brier = inst.compute_rolling_brier_score()
    if brier is not None:
        lines += [
            f"aether_trading_brier_score {round(brier, 4)}",
            f"aether_trading_brier_samples {len(inst._recent_outcomes)}",
        ]
    if inst._session_start_balance > 0.0 and inst._session_target_win > 0.0:
        lines += [
            f"aether_session_balance_usd {round(inst._current_balance, 2)}",
            f"aether_session_start_balance_usd {round(inst._session_start_balance, 2)}",
            f"aether_session_profit_usd {round(profit, 4)}",
            f"aether_session_target_win_usd {round(inst._session_target_win, 2)}",
            f"aether_session_target_balance_usd {round(target_bal, 2)}",
            f"aether_session_remaining_usd {round(rem, 2)}",
            f"aether_session_progress_pct {round(prog, 2)}",
            f"aether_session_roi_pct {round(roi, 2)}",
            f"aether_session_active_trades {inst._active_contracts_count}",
            f"aether_session_uptime_seconds {round(inst.session_uptime_seconds(), 1)}",
            f"aether_session_start_time_seconds {int(inst._session_start_time)}",
            f"aether_session_stop_win_reached {1 if inst._session_stop_win_time is not None else 0}",
        ]
    for sym, d in inst._radar.items():
        lines.extend(
            [
                f'aether_inference_prob{{symbol="{sym}"}} {round(d["prob"], 4)}',
                f'aether_inference_directional_margin{{symbol="{sym}"}} {round(d["margin"], 4)}',
                f'aether_inference_conviction{{symbol="{sym}"}} {round(d["conviction"], 4)}',
                f'aether_anti_trend_lock_active{{symbol="{sym}"}} {int(d["anti_trend_lock"])}',
                f'aether_trading_direction{{symbol="{sym}"}} {round(d["direction_num"], 1)}',
                f'aether_trading_stake_usd{{symbol="{sym}"}} {round(d["stake_usd"], 2)}',
            ]
        )
        if d["cal"] is not None:
            lines.append(f'aether_inference_calibrated{{symbol="{sym}"}} {round(d["cal"], 4)}')
        if d["edge"] is not None:
            lines.append(f'aether_inference_payoff_edge{{symbol="{sym}"}} {round(d["edge"], 4)}')

    for contract_id, (sym, direction, probability, epoch) in inst._indicator_contracts.items():
        lbl = f'symbol="{sym}",contract_id="{contract_id}"'
        lines.append(f"aether_indicator_active_trade_direction{{{lbl}}} {direction}")
        if probability is not None:
            lines.append(f"aether_indicator_active_trade_call_probability{{{lbl}}} {round(probability, 4)}")
        lines.append(f"aether_indicator_active_trade_candle_epoch{{{lbl}}} {epoch}")

    lines.extend(
        f'aether_trading_contracts_total{{symbol="{s}",direction="{d}",outcome="{o}"}} {c}'
        for (s, d, o), c in inst._contract_counts.items()
    )
    lines.extend(
        f'aether_trading_gate_verdicts_total{{symbol="{s}",verdict="{v}",reason="{r}"}} {c}'
        for (s, v, r), c in inst._gate_verdicts.items()
    )
    lines.extend(inst.runtime.render_prometheus_lines())
    return "\n".join(lines) + "\n"
