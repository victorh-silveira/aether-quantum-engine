"""Registro de resultado de contrato no RiskManager."""

from __future__ import annotations

from typing import Any

from src.domain.risk.risk_recovery_state import apply_win_to_pending_loss, log_partial_win_recovery


def apply_contract_settlement_result(
    rm: Any,
    profit: float,
    contract_id: int,
    symbol: str,
    current_tick: int = 0,
    *,
    direction: str | None = None,
) -> None:
    """Atualiza pending_loss, lucro de sessao e cluster apos liquidacao de contrato."""
    if contract_id in rm.cluster_results:
        return

    tracked = int(contract_id) in rm.active_contract_ids
    late = not tracked and int(contract_id) in rm.contract_to_symbol
    if not tracked and not late:
        return
    if late:
        rm.logger.debug("RISK: Liquidacao tardia cid=%s aplicada ao pending.", contract_id)

    recorded_stake = rm.contract_stakes.pop(int(contract_id), None)
    requested_stake = rm.contract_requested_stakes.pop(int(contract_id), None)
    rm.cluster_results[contract_id] = profit
    rm.total_session_profit += profit
    rm.last_result_tick = current_tick
    rm.record_trade_outcome(symbol, won=profit >= 0.0)

    if profit < 0:
        if requested_stake is not None and float(requested_stake) > 0.0:
            loss_amt = float(requested_stake)
        elif recorded_stake is not None:
            loss_amt = float(recorded_stake)
        else:
            loss_amt = abs(profit)
        rm.pending_loss[symbol] = rm.pending_loss.get(symbol, 0.0) + loss_amt
        rm.last_loss_stake = float(recorded_stake) if recorded_stake else loss_amt
        rm.register_symbol_loss_cooldown(symbol, direction=direction)
    else:
        recovery_profit = float(profit)
        if requested_stake is not None and recorded_stake is not None:
            downgrade_delta = max(0.0, float(requested_stake) - float(recorded_stake))
            recovery_profit = max(0.0, recovery_profit - downgrade_delta)
        apply_win_to_pending_loss(rm.pending_loss, recovery_profit)
        if log_partial_win_recovery(rm, profit) <= 0.0:
            rm.last_loss_stake = 0.0

    rm.active_contract_ids = [x for x in rm.active_contract_ids if int(x) != int(contract_id)]

    expected = rm.expected_cluster_settlements
    cluster_done = expected > 0 and len(rm.cluster_results) >= expected
    idle_done = not rm.active_contract_ids and rm.cluster_results
    if cluster_done or idle_done:
        rm._finalize_cluster()
