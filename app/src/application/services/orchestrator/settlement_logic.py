"""Lógica de liquidação e pós-processamento de contratos para o Orquestrador."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_outcomes import record_symbol_outcome
from src.application.services.deep_learning.dl_retrain import mark_force_retrain
from src.application.services.direction_loss_tracker import record_direction_outcome
from src.application.services.market_audit_log import (
    format_settlement_audit_line,
    pop_contract_audit,
)
from src.application.services.orchestrator.graceful_shutdown import graceful_shutdown
from src.application.services.orchestrator.metrics_utils import neutral_metrics
from src.application.services.orchestrator.orchestrator_atomic_state import orchestrator_atomic_state_context
from src.application.services.orchestrator.result_utils import api_settlement_label
from src.application.services.orchestrator.session_persistence_barrier import (
    consume_linear_reset_flag,
    run_linear_reset_persistence_barrier,
)
from src.application.services.orchestrator.settlement_detect import contract_payload_is_settled
from src.application.services.orchestrator.settlement_queue_ops import (
    push_to_redis_priority_queue,
)
from src.domain.risk.executed_stake_reconciliation import (
    bind_executed_stake_for_contract,
    reconcile_settlement_profit,
    resolve_executed_buy_stake,
)
from src.domain.risk.recovery_hurst_decay import reset_recovery_skip_counter_for_orch
from src.domain.risk.stop_win_target import resolve_stop_win_target


def _process_contract_outcome(orch: Any, c: dict, contract: Any, c_id: int, profit: float):
    """Atualiza o saldo da conta, registra o resultado no risk manager e computa estatísticas da sessão."""
    api_balance = c.get("balance_after")
    orch.state.balance = (
        float(api_balance)
        if api_balance is not None
        and (orch.state.balance <= 0 or abs(float(api_balance) - (orch.state.balance + profit)) <= 2.0)
        else float(orch.state.balance + profit)
    )

    sym = orch.risk_manager.contract_to_symbol.get(c_id, c.get("underlying", "UNK"))
    loss_dir = getattr(contract, "direction", None)
    dir_name = loss_dir.name if loss_dir is not None else None
    planned_stake = float(getattr(contract, "stake", 0.0) or 0.0)
    if planned_stake > 0.0:
        orch.risk_manager.contract_requested_stakes.setdefault(c_id, planned_stake)
    executed_buy = resolve_executed_buy_stake(
        c_id,
        payload=c if isinstance(c, dict) else None,
        contract=contract,
        contract_stakes=orch.risk_manager.contract_stakes,
    )
    profit = reconcile_settlement_profit(profit, executed_buy)
    bind_executed_stake_for_contract(orch.risk_manager.contract_stakes, c_id, executed_buy)
    record_symbol_outcome(orch, sym, won=profit >= 0.0)
    if dir_name:
        record_direction_outcome(sym, dir_name, won=profit >= 0.0)
    orch.risk_manager.register_result(profit, c_id, symbol=sym, current_tick=orch.tick_count, direction=dir_name)
    orch._cluster_results.append({"symbol": sym, "profit": profit})
    orch._last_result_cycle_id = orch._contract_cycle.pop(c_id, 0)
    orch._last_settlement_outcome = "WIN" if profit > 0.0 else ("LOSS" if profit < 0.0 else "FLAT")

    if profit >= 0:
        orch._session_wins += 1
    else:
        orch._session_losses += 1
        orch._last_loss_symbol = sym
        orch._last_loss_direction = dir_name or ""
        mark_force_retrain(orch, sym)

    if not orch.risk_manager.active_contract_ids:
        log_cluster_summary(orch)


def _sync_state_manager_session(orch: Any, target: float, *, increment_trades: bool) -> bool:
    """Sincroniza StateManager com saldo corrente e retorna se stop-win foi atingido."""
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is None or type(state_mgr).__name__ != "StateManager":
        return False
    if hasattr(state_mgr, "mirror_balance"):
        state_mgr.mirror_balance(float(orch.state.balance))
    else:
        state_mgr.state.current_balance = float(orch.state.balance)
    if state_mgr.state.initial_balance <= 0.0:
        state_mgr.state.initial_balance = float(orch.risk_manager.initial_bankroll)
    if state_mgr.state.daily_stop_win_target <= 0.0:
        state_mgr.state.daily_stop_win_target = float(target)
    if increment_trades:
        state_mgr.state.total_trades_today += 1
    state_mgr.check_session_limits()
    return bool(state_mgr.state.stop_win_triggered)


def check_session_limits_before_post_settlement(orch: Any) -> bool:
    """True quando o stop-win da sessao ja foi atingido antes do ciclo pos-liquidacao."""
    pnl = orch.risk_manager.total_session_profit
    target = resolve_stop_win_target(orch.config.get("risk_management", {}), orch.risk_manager.initial_bankroll)
    if target > 0 and pnl >= target:
        _sync_state_manager_session(orch, target, increment_trades=False)
        return True
    if _sync_state_manager_session(orch, target, increment_trades=False):
        return True
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is not None and type(state_mgr).__name__ == "StateManager":
        return bool(state_mgr.state.stop_win_triggered)
    return target > 0 and pnl >= target  # pragma: no cover


def _update_state_manager_and_check_stop_win(orch: Any, target: float, pnl: float) -> bool:
    """Atualiza o StateManager se disponível e retorna se o Stop Win foi ativado."""
    triggered = _sync_state_manager_session(orch, target, increment_trades=True)
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is not None and type(state_mgr).__name__ == "StateManager":
        state_mgr.save_state()
        return triggered
    return target > 0 and pnl >= target  # pragma: no cover


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
) -> None:
    """Processa liquidação, risco e persistencia sob lock atomico."""
    if result_line is not None:
        if orch._buffer_result_logs:
            orch._pending_result_logs.append(result_line)
        else:
            orch.logger.info(result_line)

    _process_contract_outcome(orch, c, contract, c_id, profit)

    if profit >= 0.0 and sum(orch.risk_manager.pending_loss.values()) <= 0.0:
        await reset_recovery_skip_counter_for_orch(orch)

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
    _, direction, edge = pop_contract_audit(orch, c_id, symbol=str(sym))
    orch.logger.info(
        "%s || API: %s (late)",
        format_settlement_audit_line(
            orch._contract_cycle.get(c_id, 0),
            outcome,
            profit,
            direction,
            str(sym),
            edge,
        ),
        api_status_raw.lower() or "-",
    )
    async with orchestrator_atomic_state_context(orch):
        _process_contract_outcome(orch, poc, None, c_id, profit)
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
    api_status_raw = (c.get("status") or "").strip()
    outcome = api_settlement_label(api_status_raw, profit)
    sym = orch.risk_manager.contract_to_symbol.get(c_id, c.get("underlying", "UNK"))
    _, direction, edge = pop_contract_audit(orch, c_id, contract=contract, symbol=str(sym))
    result_line = format_settlement_audit_line(
        orch._contract_cycle.get(c_id, 0),
        outcome,
        profit,
        direction,
        str(sym),
        edge,
    )
    async with orchestrator_atomic_state_context(orch):
        await _complete_contract_settlement(
            orch,
            c,
            contract,
            c_id,
            profit,
            result_line=result_line,
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
