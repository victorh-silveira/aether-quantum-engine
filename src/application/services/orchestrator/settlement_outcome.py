"""Processamento de resultado e sincronizacao de saldo na liquidacao."""

from __future__ import annotations

import logging
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
        if api_balance is not None
        and (orch.state.balance <= 0 or abs(float(api_balance) - (orch.state.balance + profit)) <= 2.0)
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


def sync_state_manager_session(orch: Any, target: float = 0.0, *, increment_trades: bool = True) -> bool:
    """Sincroniza StateManager com saldo corrente sem bloquear continuidade."""
    _ = target
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is None or type(state_mgr).__name__ != "StateManager":
        return False
    state_mgr.state.current_balance = float(orch.state.balance)
    if state_mgr.state.initial_balance <= 0.0:
        state_mgr.state.initial_balance = float(orch.risk_manager.initial_bankroll)
    if increment_trades:
        state_mgr.state.total_trades_today += 1
    state_mgr.save_state()
    return False


def check_session_limits_before_post_settlement(orch: Any) -> bool:
    """Modo continuo nunca interrompe o ciclo por stop-win."""
    _ = orch
    return False
