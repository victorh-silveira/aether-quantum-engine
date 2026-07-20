"""Reconciliacao de stake executada vs stake planejada em liquidacao."""

from __future__ import annotations

from typing import Any

from src.domain.risk.consensus_stake_penalty import resolve_contract_payout
from src.domain.risk.risk_recovery_state import apply_win_to_pending_loss, log_partial_win_recovery


_MAX_FRACTIONAL_PAYOFF_RESIDUAL_CENTS = 10


def resolve_executed_buy_stake(
    contract_id: int,
    *,
    payload: dict[str, Any] | None = None,
    contract: Any | None = None,
    contract_stakes: dict[int, float] | None = None,
) -> float:
    """Resolve valor debitado na compra priorizando buy_price confirmado pela API."""
    if isinstance(payload, dict):
        for key in ("buy_price", "purchase_price", "amount"):
            raw = payload.get(key)
            if raw is not None:
                value = float(raw)
                if value > 0.0:
                    return value
    if contract is not None:
        buy = getattr(contract, "buy_price", None)
        if buy is not None:
            value = float(buy)
            if value > 0.0:
                return value
        stake = getattr(contract, "stake", None)
        if stake is not None:
            value = float(stake)
            if value > 0.0:
                return value
    if isinstance(contract_stakes, dict):
        stored = contract_stakes.get(int(contract_id))
        if stored is not None:
            value = float(stored)
            if value > 0.0:
                return value
    return 0.0


def reconcile_settlement_profit(api_profit: float, executed_buy: float) -> float:
    """Alinha P&L de perda ao valor efetivamente debitado na compra."""
    profit = float(api_profit)
    stake = max(0.0, float(executed_buy))
    if profit < 0.0 and stake > 0.0:
        return -stake
    return profit


def bind_executed_stake_for_contract(
    contract_stakes: dict[int, float],
    contract_id: int,
    executed_buy: float,
) -> None:
    """Sobrescreve stake planejada pela compra confirmada antes do registro de resultado."""
    stake = max(0.0, float(executed_buy))
    if stake > 0.0:
        contract_stakes[int(contract_id)] = stake


def domain_expected_win_profit(executed_stake: float, payout_rate: float) -> float:
    """Payoff de dominio para WIN: stake executada multiplicada pelo payout b."""
    stake = max(0.0, float(executed_stake))
    rate = max(0.0, float(payout_rate))
    return stake * rate


def fractional_payoff_residual_cents(api_profit: float, executed_stake: float, payout_rate: float) -> float:
    """Diferenca explicita com sub-centavos entre payoff de dominio e P&L real da API."""
    profit = float(api_profit)
    stake = float(executed_stake)
    if profit <= 0.0 or stake <= 0.0:
        return 0.0
    expected = domain_expected_win_profit(stake, payout_rate)
    residual = round(profit - expected, 4)
    if abs(residual) > 0.10:
        return 0.0
    return residual


def apply_fractional_payoff_residual_to_pending(
    pending_loss: dict[str, float],
    symbol: str,
    residual: float,
) -> None:
    """Ajusta pending_loss com residuo centavado de payoff fracionado."""
    delta = float(residual)
    if delta == 0.0:
        return
    current = float(pending_loss.get(symbol, 0.0))
    if delta < 0.0 and abs(delta) <= 0.10 and current <= 0.0:
        pending_loss.pop(symbol, None)
        return
    updated = max(0.0, current - delta)
    if updated <= 0.0:
        pending_loss.pop(symbol, None)
    else:
        pending_loss[symbol] = updated


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
        if recorded_stake is not None and float(recorded_stake) > 0.0:
            payout_rate = resolve_contract_payout(None, getattr(rm, "risk_params", None))
            residual = fractional_payoff_residual_cents(profit, float(recorded_stake), payout_rate)
            apply_fractional_payoff_residual_to_pending(rm.pending_loss, symbol, residual)
        if log_partial_win_recovery(rm, profit) <= 0.0:
            rm.last_loss_stake = 0.0

    rm.active_contract_ids = [x for x in rm.active_contract_ids if int(x) != int(contract_id)]

    expected = rm.expected_cluster_settlements
    cluster_done = expected > 0 and len(rm.cluster_results) >= expected
    idle_done = not rm.active_contract_ids and rm.cluster_results
    if cluster_done or idle_done:
        rm._finalize_cluster()
