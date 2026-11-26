"""Logica de liquidacao e pos-processamento de contratos no modo Laya continuo."""

from __future__ import annotations

from typing import Any

from src.application.services.log_dedupe import log_warning_if_changed
from src.application.services.orchestrator.orchestrator_atomic_state import orchestrator_atomic_state_context
from src.application.services.orchestrator.result_utils import api_settlement_label
from src.application.services.orchestrator.settlement_audit import record_settlement_audit
from src.application.services.orchestrator.settlement_helpers import (
    contract_payload_is_settled,
    emit_contract_settlement_log as _emit_contract_settlement_log,
    log_cluster_summary,
    process_settlement_outcome as _process_contract_outcome,
    settlement_lock,
    settlement_seen,
)
from src.application.services.orchestrator.settlement_outcome import (
    check_session_limits_before_post_settlement,
    sync_state_manager_session,
)
from src.application.services.orchestrator.settlement_queue_ops import push_to_redis_priority_queue


_sync_state_manager_session = sync_state_manager_session

__all__ = (
    "check_session_limits_before_post_settlement",
    "log_cluster_summary",
    "process_contract_settlement",
    "process_late_settlement_from_payload",
)


def _format_settlement_line(orch: Any, c_id: int, outcome: str, profit: float, sym: str, direction: str) -> str:
    """Formata linha padronizada de liquidacao para o log do motor."""
    pnl = float(getattr(orch.risk_manager, "total_session_profit", 0.0) or 0.0)
    bal = float(getattr(orch.state, "balance", 0.0) or 0.0)
    return (
        f"SETTLE | {sym} {direction} | {outcome} pnl={profit:+.2f} | sessao={pnl:+.2f} | banca=${bal:.2f} | cid={c_id}"
    )


async def _complete_settlement(
    orch: Any, payload: dict, contract: Any, c_id: int, profit: float, *, late: bool = False
) -> None:
    """Aplica liquidacao, atualiza saldo/telemetria e agenda proximo ciclo."""
    sym = str(
        getattr(contract, "symbol", None)
        or orch.risk_manager.contract_to_symbol.get(c_id)
        or payload.get("underlying")
        or "R_10"
    )
    dir_obj = getattr(contract, "direction", None)
    direction = dir_obj.name if dir_obj is not None else "CALL"
    outcome = api_settlement_label((payload.get("status") or "").strip(), profit)
    contract_cycle = int(orch._contract_cycle.get(c_id, 0) or getattr(orch, "_last_result_cycle_id", 0) or 0)
    async with orchestrator_atomic_state_context(orch):
        _process_contract_outcome(orch, payload, contract, c_id, profit, audit_direction=direction)
        line = _format_settlement_line(orch, c_id, outcome, profit, sym, direction)
        if late:
            line = f"{line} (late)"
        _emit_contract_settlement_log(orch, line, c_id=c_id, sym=sym, cycle_id=contract_cycle or None)
        sync_state_manager_session(orch, 0.0, increment_trades=True)
        if not orch.state.active_contracts and orch.running:
            orch.schedule_trading_cycle_after_settlement()
        await orch._persist_full_state_unlocked()


async def process_late_settlement_from_payload(orch: Any, poc: dict) -> None:
    """Liquida contrato ausente de active_contracts mas rastreado em risco."""
    if not contract_payload_is_settled(poc):
        return
    c_id = poc.get("contract_id")
    if c_id is None:
        return
    c_id = int(c_id)
    async with settlement_lock(orch):
        if c_id in settlement_seen(orch):
            return
        contract = await orch.state.finalize_contract(c_id)
        if contract is None:
            tracked = c_id in orch.risk_manager.contract_to_symbol or c_id in orch.risk_manager.active_contract_ids
            if not tracked:
                return
        else:
            sym = str(getattr(contract, "symbol", None) or poc.get("underlying") or "R_10")
            await record_settlement_audit(orch, poc, contract, sym)
        await _complete_settlement(orch, poc, contract, c_id, float(poc.get("profit", 0.0)), late=contract is None)
        settlement_seen(orch).add(c_id)


async def process_contract_settlement(orch: Any, data: dict) -> None:
    """Lida com mensagem proposal_open_contract liquidada e atualiza estado."""
    if "proposal_open_contract" not in data:
        return
    c = data["proposal_open_contract"]
    if not contract_payload_is_settled(c):
        return
    c_id = c.get("contract_id")
    if c_id is None:
        return
    c_id = int(c_id)
    if not orch.ws.is_running:
        log_warning_if_changed(
            orch,
            orch.logger,
            f"settle_broker_offline:{c_id}",
            "offline",
            "SETTLE.settle_broker: Broker offline. Enfileirando contrato %d no Redis.",
            c_id,
        )
        await push_to_redis_priority_queue(orch, data)
        return
    async with settlement_lock(orch):
        if c_id in settlement_seen(orch):
            return
        contract = await orch.state.finalize_contract(c_id)
        if contract is None:
            tracked = c_id in orch.risk_manager.contract_to_symbol or c_id in orch.risk_manager.active_contract_ids
            if not tracked:
                return
        else:
            sym = str(getattr(contract, "symbol", None) or c.get("underlying") or "R_10")
            await record_settlement_audit(orch, c, contract, sym)
        await _complete_settlement(orch, c, contract, c_id, float(c.get("profit", 0.0)), late=contract is None)
        settlement_seen(orch).add(c_id)
