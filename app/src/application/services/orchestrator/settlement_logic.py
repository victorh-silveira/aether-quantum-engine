"""Lógica de liquidação e pós-processamento de contratos para o Orquestrador."""

from __future__ import annotations

from typing import Any

from src.application.services.indicator_order_claims import release_order_symbol_claim
from src.application.services.log_dedupe import log_warning_if_changed
from src.application.services.market_audit_contract import broker_price_audit
from src.application.services.market_audit_log import (
    format_settlement_audit_line,
    pop_contract_audit,
    resolve_settlement_tag,
    resolve_stake_audit_context,
)
from src.application.services.orchestrator.graceful_shutdown import graceful_shutdown
from src.application.services.orchestrator.orchestrator_atomic_state import orchestrator_atomic_state_context
from src.application.services.orchestrator.result_utils import api_settlement_label
from src.application.services.orchestrator.session_persistence_barrier import (
    consume_linear_reset_flag,
    run_linear_reset_persistence_barrier,
)
from src.application.services.orchestrator.settlement_audit import record_settlement_audit
from src.application.services.orchestrator.settlement_cluster_log import log_cluster_summary
from src.application.services.orchestrator.settlement_dedupe import settlement_lock, settlement_seen
from src.application.services.orchestrator.settlement_detect import contract_payload_is_settled
from src.application.services.orchestrator.settlement_loss_quarantine import handle_settlement_loss_quarantine
from src.application.services.orchestrator.settlement_outcome import (
    check_session_limits_before_post_settlement,
    sync_state_manager_session,
    update_state_manager_and_check_stop_win,
)
from src.application.services.orchestrator.settlement_process import (
    emit_contract_settlement_log as _emit_contract_settlement_log,
    process_settlement_outcome as _process_contract_outcome,
)
from src.application.services.orchestrator.settlement_queue_ops import push_to_redis_priority_queue
from src.application.services.pullback_recovery_claim import release_recovery
from src.application.services.recovery_hurst_store import reset_recovery_skip_counter_for_orch
from src.domain.risk.stop_win_target import resolve_stop_win_target


_sync_state_manager_session = sync_state_manager_session

__all__ = (
    "check_session_limits_before_post_settlement",
    "log_cluster_summary",
    "process_contract_settlement",
    "process_late_settlement_from_payload",
)


async def _finalize_settlement_persistence(orch: Any) -> bool:
    """Aplica barreira pos-reset linear ou persistencia padrao ao encerrar liquidacao."""
    if consume_linear_reset_flag(orch):
        await run_linear_reset_persistence_barrier(orch)
        return True
    await orch._persist_full_state_unlocked()
    return False


def _format_audit_row(
    orch: Any, c_id: int, outcome: str, profit: float, direction: Any, sym: str, edge: float, linear_before: int
) -> str:
    """Monta a linha padronizada de auditoria para o contrato liquidado."""
    stake_audit = resolve_stake_audit_context(orch.risk_manager)
    learn = str(getattr(orch, "_last_loss_clf_learn", "") or "").strip()
    pnl = float(getattr(orch.risk_manager, "total_session_profit", 0.0) or 0.0)
    target = resolve_stop_win_target(orch.config.get("risk_management", {}), orch.risk_manager.initial_bankroll)
    c_cycle = int(orch._contract_cycle.get(c_id, 0) or getattr(orch, "_last_result_cycle_id", 0) or 0)
    pending = float(stake_audit.get("pending", 0.0))
    tag = resolve_settlement_tag(profit=profit, linear_before=linear_before, pending_after=pending)
    mode = str(getattr(orch, "_pullback_contract_modes", {}).get(c_id) or stake_audit.get("mode_tag") or "")
    return format_settlement_audit_line(
        c_cycle,
        outcome,
        profit,
        direction,
        sym,
        edge,
        settlement_tag=tag,
        pending=pending,
        linear=int(getattr(orch.risk_manager, "consecutive_losses_linear", 0) or 0),
        mode_tag=mode,
        recovery_infeasible=bool(stake_audit.get("recovery_infeasible", False)),
        learn_detail=learn or None,
        session_pnl=pnl,
        target_pnl=target,
        bankroll=float(getattr(orch.state, "balance", 0.0) or (orch.risk_manager.initial_bankroll + pnl)),
    )


async def _complete_contract_settlement(
    orch: Any,
    c: dict,
    contract: Any,
    c_id: int,
    profit: float,
    *,
    result_line: str | None = None,
    audit_direction: str | None = None,
    audit_raw_prob: float | None = None,
    resolved_meta: dict[str, Any] | None = None,
) -> None:
    """Processa liquidação, risco e persistencia sob lock atomico."""
    contract_cycle = int(orch._contract_cycle.get(c_id, 0) or getattr(orch, "_last_result_cycle_id", 0) or 0)
    _process_contract_outcome(
        orch, c, contract, c_id, profit, audit_direction=audit_direction, audit_raw_prob=audit_raw_prob
    )
    pnl = float(getattr(orch.risk_manager, "total_session_profit", 0.0) or 0.0)
    target = resolve_stop_win_target(orch.config.get("risk_management", {}), orch.risk_manager.initial_bankroll)
    if result_line is None and isinstance(resolved_meta, dict):
        result_line = _format_audit_row(
            orch,
            c_id,
            str(resolved_meta.get("outcome") or ""),
            profit,
            resolved_meta.get("direction"),
            str(resolved_meta.get("symbol") or ""),
            float(resolved_meta.get("edge") or 0.0),
            int(resolved_meta.get("linear_before", 0) or 0),
        )
    if result_line is not None:
        price_audit = broker_price_audit(
            c, audit_direction or getattr(getattr(contract, "direction", None), "name", None)
        )
        line = f"{result_line} | {price_audit}" if price_audit else result_line
        _emit_contract_settlement_log(
            orch,
            line,
            c_id=c_id,
            sym=resolved_meta.get("symbol") if isinstance(resolved_meta, dict) else None,
            cycle_id=contract_cycle or getattr(orch, "_last_result_cycle_id", None),
        )
    getattr(orch, "_pullback_contract_modes", {}).pop(c_id, None)
    sym_resolved = str(resolved_meta.get("symbol") or "") if isinstance(resolved_meta, dict) else ""
    await release_order_symbol_claim(orch, sym_resolved)
    await handle_settlement_loss_quarantine(orch, resolved_meta, contract, c_id, profit)
    if profit >= 0.0 and sum(orch.risk_manager.pending_loss.values()) <= 0.0:
        await reset_recovery_skip_counter_for_orch(orch)

    stop_win_triggered = update_state_manager_and_check_stop_win(orch, target, pnl)

    if stop_win_triggered:
        if getattr(orch, "business_metrics", None):
            orch.business_metrics.mark_stop_win()
        orch.logger.debug(
            "[C%04d] STOP_WIN | pnl_sessao=$%+.2f | alvo=$%.2f",
            orch._last_result_cycle_id,
            pnl,
            target,
        )
        orch.shutdown_reason = "stop_win"
        await orch.state.set_trading(value=False)
        await orch._persist_full_state_unlocked()
        await graceful_shutdown(orch, fast_path=True)
        return
    if not orch.state.active_contracts and orch.running:
        orch.schedule_trading_cycle_after_settlement()

    await _finalize_settlement_persistence(orch)


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
        if contract:
            await _process_confirmed_settlement(orch, {"proposal_open_contract": poc}, contract)
            await release_recovery(orch, c_id)
            settlement_seen(orch).add(c_id)
            return
        tracked = c_id in orch.risk_manager.contract_to_symbol or c_id in orch.risk_manager.active_contract_ids
        if not tracked:
            return
        await _process_late_settlement_once(orch, poc, c_id)
        await release_recovery(orch, c_id)
        settlement_seen(orch).add(c_id)


async def _process_late_settlement_once(orch: Any, poc: dict, c_id: int) -> None:
    """Aplica uma liquidacao tardia ja reservada pelo lock de contrato."""
    profit = float(poc.get("profit", 0.0))
    api_status_raw = (poc.get("status") or "").strip()
    outcome = api_settlement_label(api_status_raw, profit)
    contract_cycle = int(orch._contract_cycle.get(c_id, 0) or getattr(orch, "_last_result_cycle_id", 0) or 0)
    sym = orch.risk_manager.contract_to_symbol.get(c_id, poc.get("underlying", "UNK"))
    _, direction, edge, z_score, raw_prob = pop_contract_audit(orch, c_id)
    linear_before = int(getattr(orch.risk_manager, "consecutive_losses_linear", 0) or 0)
    async with orchestrator_atomic_state_context(orch):
        _process_contract_outcome(orch, poc, None, c_id, profit, audit_direction=direction, audit_raw_prob=raw_prob)
        audit_line = _format_audit_row(orch, c_id, outcome, profit, direction, str(sym), edge, linear_before)
        _emit_contract_settlement_log(
            orch,
            f"{audit_line} || API: {api_status_raw.lower() or '-'} (late)",
            c_id=c_id,
            sym=str(sym),
            cycle_id=contract_cycle or getattr(orch, "_last_result_cycle_id", None),
        )
        orch._last_loss_clf_learn = None
        getattr(orch, "_pullback_contract_modes", {}).pop(c_id, None)
        await release_order_symbol_claim(orch, str(sym))
        if profit >= 0.0 and sum(orch.risk_manager.pending_loss.values()) <= 0.0:
            await reset_recovery_skip_counter_for_orch(orch)
        if not orch.state.active_contracts and orch.running:
            orch.schedule_trading_cycle_after_settlement()
        await _finalize_settlement_persistence(orch)


async def _process_confirmed_settlement(orch: Any, data: dict, contract: Any) -> None:
    """Aplica o resultado confirmado e reconciliado do contrato ao RiskManager e estado."""
    c = data["proposal_open_contract"]
    c_id = int(c["contract_id"])
    profit = float(c.get("profit", 0.0))
    outcome = api_settlement_label((c.get("status") or "").strip(), profit)
    sym = (
        getattr(contract, "symbol", None)
        or orch.risk_manager.contract_to_symbol.get(c_id)
        or c.get("underlying")
        or "UNK"
    )
    _, direction, edge, z_score, raw_prob = pop_contract_audit(orch, c_id)
    await record_settlement_audit(orch, c, contract, str(sym), signal_prob=raw_prob)
    linear_before = int(getattr(orch.risk_manager, "consecutive_losses_linear", 0) or 0)
    async with orchestrator_atomic_state_context(orch):
        await _complete_contract_settlement(
            orch,
            c,
            contract,
            c_id,
            profit,
            audit_direction=direction,
            audit_raw_prob=raw_prob,
            resolved_meta={
                "outcome": outcome,
                "direction": direction,
                "symbol": sym,
                "edge": edge,
                "linear_before": linear_before,
            },
        )


async def process_contract_settlement(orch: Any, data: dict):
    """Lida com a mensagem de liquidação, atualiza saldo, risco e logs."""
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
        if not contract:
            tracked = c_id in orch.risk_manager.contract_to_symbol or c_id in orch.risk_manager.active_contract_ids
            if not tracked:
                return
            await _process_late_settlement_once(orch, c, c_id)
        else:
            await _process_confirmed_settlement(orch, data, contract)
        await release_recovery(orch, c_id)
        settlement_seen(orch).add(c_id)
