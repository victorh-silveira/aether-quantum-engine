"""Lógica de liquidação e pós-processamento de contratos para o Orquestrador."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate_starvation import (
    reset_quality_skipped_cycles_counter_for_orch,
)
from src.application.services.market_audit_log import (
    format_settlement_audit_line,
    pop_contract_audit,
    resolve_settlement_tag,
    resolve_stake_audit_context,
)
from src.application.services.meta_payoff_shadow import record_meta_payoff_shadow_pair
from src.application.services.orchestrator.graceful_shutdown import graceful_shutdown
from src.application.services.orchestrator.metrics_utils import neutral_metrics
from src.application.services.orchestrator.orchestrator_atomic_state import orchestrator_atomic_state_context
from src.application.services.orchestrator.result_utils import api_settlement_label
from src.application.services.orchestrator.session_persistence_barrier import (
    consume_linear_reset_flag,
    run_linear_reset_persistence_barrier,
)
from src.application.services.orchestrator.settlement_detect import contract_payload_is_settled
from src.application.services.orchestrator.settlement_outcome import (
    check_session_limits_before_post_settlement,
    process_contract_outcome,
    sync_state_manager_session,
    update_state_manager_and_check_stop_win,
)
from src.application.services.orchestrator.settlement_queue_ops import (
    push_to_redis_priority_queue,
)
from src.domain.risk.recovery_hurst_decay import reset_recovery_skip_counter_for_orch
from src.domain.risk.stop_win_target import resolve_stop_win_target


__all__ = [
    "check_session_limits_before_post_settlement",
    "log_cluster_summary",
    "process_contract_settlement",
    "process_late_settlement_from_payload",
]

_sync_state_manager_session = sync_state_manager_session
_update_state_manager_and_check_stop_win = update_state_manager_and_check_stop_win


def _process_contract_outcome(
    orch: Any,
    c: dict,
    contract: Any,
    c_id: int,
    profit: float,
    *,
    audit_direction: str | None = None,
    audit_raw_prob: float | None = None,
):
    """Atualiza o saldo da conta, registra o resultado no risk manager e computa estatísticas da sessão."""
    return process_contract_outcome(
        orch,
        c,
        contract,
        c_id,
        profit,
        audit_direction=audit_direction,
        audit_raw_prob=audit_raw_prob,
        log_cluster_summary=log_cluster_summary,
    )


async def _finalize_settlement_persistence(orch: Any) -> bool:
    """Aplica barreira pos-reset linear ou persistencia padrao ao encerrar liquidacao."""
    if consume_linear_reset_flag(orch):
        await run_linear_reset_persistence_barrier(orch)
        return True
    await orch._persist_full_state_unlocked()
    return False


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
) -> None:
    """Processa liquidação, risco e persistencia sob lock atomico."""
    if result_line is not None:
        if orch._buffer_result_logs:
            orch._pending_result_logs.append(result_line)
        else:
            orch.logger.info(result_line)

    _process_contract_outcome(
        orch,
        c,
        contract,
        c_id,
        profit,
        audit_direction=audit_direction,
        audit_raw_prob=audit_raw_prob,
    )

    if profit >= 0.0 and sum(orch.risk_manager.pending_loss.values()) <= 0.0:
        await reset_recovery_skip_counter_for_orch(orch)
        await reset_quality_skipped_cycles_counter_for_orch(orch)

    pnl = orch.risk_manager.total_session_profit
    target = resolve_stop_win_target(orch.config.get("risk_management", {}), orch.risk_manager.initial_bankroll)

    stop_win_triggered = _update_state_manager_and_check_stop_win(orch, target, pnl)

    if stop_win_triggered:
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
    profit = float(poc.get("profit", 0.0))
    api_status_raw = (poc.get("status") or "").strip()
    outcome = api_settlement_label(api_status_raw, profit)
    sym = orch.risk_manager.contract_to_symbol.get(c_id, poc.get("underlying", "UNK"))
    _, direction, edge, z_score, raw_prob = pop_contract_audit(orch, c_id, symbol=str(sym))
    record_meta_payoff_shadow_pair(z_score=z_score, profit=profit, orch=orch)
    linear_before = int(getattr(orch.risk_manager, "consecutive_losses_linear", 0) or 0)
    stake_audit = resolve_stake_audit_context(orch.risk_manager)
    orch.logger.info(
        "%s || API: %s (late)",
        format_settlement_audit_line(
            orch._contract_cycle.get(c_id, 0),
            outcome,
            profit,
            direction,
            str(sym),
            edge,
            settlement_tag=resolve_settlement_tag(profit=profit, linear_before=linear_before),
            pending=float(stake_audit.get("pending", 0.0)),
            linear=int(stake_audit.get("linear", linear_before)),
            mode_tag=str(stake_audit.get("mode_tag") or ""),
            recovery_infeasible=bool(stake_audit.get("recovery_infeasible", False)),
        ),
        api_status_raw.lower() or "-",
    )
    async with orchestrator_atomic_state_context(orch):
        _process_contract_outcome(
            orch,
            poc,
            None,
            c_id,
            profit,
            audit_direction=direction,
            audit_raw_prob=raw_prob,
        )
        if profit >= 0.0 and sum(orch.risk_manager.pending_loss.values()) <= 0.0:
            await reset_recovery_skip_counter_for_orch(orch)
            await reset_quality_skipped_cycles_counter_for_orch(orch)
        if not orch.state.active_contracts and orch.running:
            orch.schedule_trading_cycle_after_settlement()
        await _finalize_settlement_persistence(orch)


async def _process_confirmed_settlement(orch: Any, data: dict, contract: Any) -> None:
    """Aplica o resultado confirmado e reconciliado do contrato ao RiskManager e estado."""
    c = data["proposal_open_contract"]
    c_id = int(c["contract_id"])
    profit = float(c.get("profit", 0.0))
    api_status_raw = (c.get("status") or "").strip()
    outcome = api_settlement_label(api_status_raw, profit)
    sym = orch.risk_manager.contract_to_symbol.get(c_id, c.get("underlying", "UNK"))
    _, direction, edge, z_score, raw_prob = pop_contract_audit(orch, c_id, contract=contract, symbol=str(sym))
    record_meta_payoff_shadow_pair(z_score=z_score, profit=profit, orch=orch)
    linear_before = int(getattr(orch.risk_manager, "consecutive_losses_linear", 0) or 0)
    stake_audit = resolve_stake_audit_context(orch.risk_manager)
    result_line = format_settlement_audit_line(
        orch._contract_cycle.get(c_id, 0),
        outcome,
        profit,
        direction,
        str(sym),
        edge,
        settlement_tag=resolve_settlement_tag(profit=profit, linear_before=linear_before),
        pending=float(stake_audit.get("pending", 0.0)),
        linear=int(stake_audit.get("linear", linear_before)),
        mode_tag=str(stake_audit.get("mode_tag") or ""),
        recovery_infeasible=bool(stake_audit.get("recovery_infeasible", False)),
    )
    async with orchestrator_atomic_state_context(orch):
        await _complete_contract_settlement(
            orch,
            c,
            contract,
            c_id,
            profit,
            result_line=result_line,
            audit_direction=direction,
            audit_raw_prob=raw_prob,
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
        orch.logger.warning("SETTLE: Broker offline. Enfileirando contrato %d no Redis.", c_id)
        await push_to_redis_priority_queue(orch, data)
        return

    contract = await orch.state.finalize_contract(c_id)
    if not contract:
        return

    await _process_confirmed_settlement(orch, data, contract)


def log_cluster_summary(orch: Any):
    """Emite resumo de performance do cluster encerrado."""
    orch.logger.debug(
        "[C%04d] BANCA FINAL: $%.2f | ACUMULADO: %dW / %dL",
        orch._last_result_cycle_id,
        orch.state.balance,
        orch._session_wins,
        orch._session_losses,
    )
    orch.logger.debug("")
    orch._cluster_results = []
    orch._last_anchor_metrics = neutral_metrics()
