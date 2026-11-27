"""Processamento de resultado e sincronizacao de saldo na liquidacao."""

from __future__ import annotations

import logging
import math
from typing import Any


logger = logging.getLogger("AETH")


def process_contract_outcome(
    orch: Any,
    c: dict[str, Any],
    contract: Any,
    c_id: int,
    profit: float,
    *,
    audit_direction: str | None = None,
    audit_raw_prob: float | None = None,
    log_cluster_summary: Any = None,
) -> None:
    """Atualiza saldo, RiskManager e telemetria apos liquidacao do contrato."""
    _ = audit_raw_prob
    api_balance = c.get("balance_after")
    orch.state.balance = (
        float(api_balance)
        if api_balance is not None and math.isfinite(float(api_balance)) and float(api_balance) > 0.0
        else float(orch.state.balance + profit)
    )
    sym = orch.risk_manager.contract_to_symbol.get(c_id, c.get("underlying", "R_10"))
    loss_dir = getattr(contract, "direction", None)
    dir_name = loss_dir.name if loss_dir is not None else (str(audit_direction) if audit_direction else "CALL")
    executed_buy = float(
        orch.risk_manager.contract_stakes.get(c_id)
        or getattr(contract, "buy_price", 0.0)
        or getattr(contract, "stake", 0.0)
        or 0.0
    )
    orch.risk_manager.on_trade_settled(c_id, float(profit), symbol=str(sym))
    orch._cluster_results.append({"symbol": sym, "profit": profit})
    orch._last_result_cycle_id = orch._contract_cycle.pop(c_id, 0)
    orch._last_settlement_outcome = "WIN" if profit > 0.0 else ("LOSS" if profit < 0.0 else "FLAT")
    if profit >= 0.0:
        orch._session_wins += 1
    else:
        orch._session_losses += 1
    bm = getattr(orch, "business_metrics", None)
    if bm is not None:
        bm.record_trade(
            symbol=str(sym),
            direction=dir_name,
            contract_id=c_id,
            profit=float(profit),
            stake=executed_buy,
            won=bool(profit >= 0.0),
        )
        bm.update_balance(float(orch.state.balance))
        bm.update_active_contracts_count(len(orch.risk_manager.active_contract_ids))
    if not orch.risk_manager.active_contract_ids and callable(log_cluster_summary):
        log_cluster_summary(orch)


def _sync_realized_pnl(orch: Any, state_mgr: Any) -> None:
    """Sincroniza lucro realizado da sessao sem misturar creditos externos de saldo."""
    rm = getattr(orch, "risk_manager", None)
    pnl = getattr(rm, "total_session_profit", None)
    if isinstance(pnl, (int, float)) and not isinstance(pnl, bool):
        state_mgr.state.realized_session_pnl = float(pnl)


def sync_state_manager_session(orch: Any, target: float = 0.0, *, increment_trades: bool = True) -> bool:
    """Sincroniza StateManager com saldo corrente e avalia stop-win."""
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is None or type(state_mgr).__name__ != "StateManager":
        return False
    state_mgr.state.current_balance = float(orch.state.balance)
    if state_mgr.state.initial_balance <= 0.0:
        state_mgr.state.initial_balance = float(orch.risk_manager.initial_bankroll)
    if target > 0.0:
        state_mgr.state.daily_stop_win_target = float(target)
    if increment_trades:
        state_mgr.state.total_trades_today += 1
    _sync_realized_pnl(orch, state_mgr)
    triggered = bool(state_mgr.check_session_limits())
    state_mgr.save_state()
    if increment_trades and hasattr(orch, "logger"):
        wins = int(getattr(orch, "_session_wins", 0) or 0)
        losses = int(getattr(orch, "_session_losses", 0) or 0)
        total = wins + losses
        if total > 0:
            wr = (wins / total) * 100.0
            pnl = float(state_mgr.state.session_profit)
            tgt = float(state_mgr.state.daily_stop_win_target)
            prog = min(100.0, max(0.0, (pnl / tgt) * 100.0)) if tgt > 0.0 else 0.0
            pending = float(getattr(getattr(orch, "risk_manager", None), "pending_loss", 0.0) or 0.0)
            orch.logger.info(
                "SESSION_STATS | W=%d L=%d WR=%.2f%% | PnL=%+.2f | meta=%.2f%% | pending=$%.2f",
                wins,
                losses,
                wr,
                pnl,
                prog,
                pending,
            )
    return triggered


def check_session_limits_before_post_settlement(orch: Any) -> bool:
    """Retorna True quando o lucro da sessao atinge o stop-win configurado."""
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is None or not hasattr(state_mgr, "check_session_limits"):
        return False
    state_mgr.state.current_balance = float(
        getattr(getattr(orch, "state", None), "balance", state_mgr.state.current_balance)
    )
    _sync_realized_pnl(orch, state_mgr)
    if not state_mgr.check_session_limits():
        return False
    orch.logger.info(
        "STOP_WIN | Meta atingida | lucro=$%.2f >= alvo=$%.2f | banca=$%.2f | encerrando sessao",
        state_mgr.state.session_profit,
        state_mgr.state.daily_stop_win_target,
        state_mgr.state.current_balance,
    )
    return True
