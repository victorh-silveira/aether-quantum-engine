"""Adocao e registro de contratos executados no estado do orquestrador."""

from __future__ import annotations

from typing import Any

from src.application.services.market_audit_log import resolve_predicted_edge, store_contract_audit
from src.domain.models.trade import TradeDirection


async def adopt_executed_contract(
    executor: Any,
    contract: Any,
    *,
    symbol: str,
    direction: TradeDirection,
    metrics: dict[str, Any],
    requested_stake: float,
    order_n: int,
) -> None:
    """Registra contrato aberto no orquestrador apos compra confirmada."""
    orch = executor.orch
    executed_stake = float(getattr(contract, "buy_price", 0.0) or requested_stake)
    cid = int(contract.contract_id)
    orch.risk_manager.record_contract_stake(cid, executed_stake, requested=requested_stake)
    orch.risk_manager.active_contract_ids.append(cid)
    await orch.state.add_contract(contract)
    orch._contract_cycle[cid] = int(orch._active_cycle_id)
    store_contract_audit(
        orch,
        cid,
        symbol=symbol,
        direction=direction.name,
        edge=resolve_predicted_edge(metrics),
    )
    executor._log_exec(
        symbol,
        direction,
        requested_stake,
        metrics,
        order_n=order_n,
        contract_id=cid,
    )
