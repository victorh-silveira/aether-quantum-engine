"""Reconciliacao de stake executada vs stake planejada em liquidacao."""

from __future__ import annotations

from typing import Any


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
