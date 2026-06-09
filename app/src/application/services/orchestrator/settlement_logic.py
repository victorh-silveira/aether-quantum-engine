"""Lógica de liquidação e pós-processamento de contratos para o Orquestrador."""

from typing import Any

from src.application.services.deep_learning.dl_outcomes import record_symbol_outcome
from src.application.services.deep_learning.dl_retrain import mark_force_retrain
from src.application.services.orchestrator.metrics_utils import neutral_metrics
from src.application.services.orchestrator.result_utils import api_settlement_label
from src.application.services.orchestrator.settlement_detect import contract_payload_is_settled
from src.application.services.orchestrator.stop_win_target import resolve_stop_win_target


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
    contract = await orch.state.finalize_contract(c_id)
    if not contract:
        return

    profit = float(c.get("profit", 0.0))
    api_status_raw = (c.get("status") or "").strip()
    outcome = api_settlement_label(api_status_raw, profit)

    result_line = (
        f"[C{orch._contract_cycle.get(c_id, 0):04d}] STATUS: {outcome} || "
        f"P&L: ${profit:+.2f} || API: {api_status_raw.lower() or '-'}"
    )

    if orch._buffer_result_logs:
        orch._pending_result_logs.append(result_line)
    else:
        orch.logger.info(result_line)

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
    record_symbol_outcome(orch, sym, won=profit >= 0.0)
    orch.risk_manager.register_result(profit, c_id, symbol=sym, current_tick=orch.tick_count, direction=dir_name)
    orch._cluster_results.append({"symbol": sym, "profit": profit})
    orch._last_result_cycle_id = orch._contract_cycle.pop(c_id, 0)

    if profit >= 0:
        orch._session_wins += 1
    else:
        orch._session_losses += 1
        orch._last_loss_symbol = sym
        orch._last_loss_direction = dir_name or ""
        mark_force_retrain(orch, sym)

    if not orch.risk_manager.active_contract_ids:
        log_cluster_summary(orch)

    pnl = orch.risk_manager.total_session_profit
    target = resolve_stop_win_target(orch.config.get("risk_management", {}), orch.risk_manager.initial_bankroll)
    if target > 0 and pnl >= target:
        orch.logger.debug("[C%04d] STOP_WIN | pnl_sessao=$%+.2f | alvo=$%.2f", orch._last_result_cycle_id, pnl, target)
        orch.shutdown_reason = "stop_win"
        orch.running = False
        await orch.state.set_trading(value=False)
    elif not orch.state.active_contracts and orch.running:
        orch.schedule_trading_cycle_after_settlement()

    await orch._save_full_state()


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
