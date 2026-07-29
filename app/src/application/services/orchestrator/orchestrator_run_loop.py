"""Ciclo de vida, persistencia e sessao do Orquestrador."""

import asyncio
import time
from typing import Any

from src.application.services.orchestrator.graceful_shutdown import close_infrastructure_connections
from src.application.services.orchestrator.orchestrator_data_signature import (
    get_data_state_signature,
    seconds_until_next_signature_boundary,
)
from src.application.services.orchestrator.orchestrator_persistence import save_full_state
from src.application.services.orchestrator.orchestrator_settlement_queue import (
    SettlementOrphanCleaner,
    start_settlement_worker,
)
from src.application.services.orchestrator.post_settlement_resilience import (
    recover_post_settlement_loop_transparently,
)
from src.application.services.orchestrator.reconnect_cycle_release import release_trading_cycle_after_reconnect
from src.application.services.orchestrator.settlement_reconciliation import reconcile_after_ws_recovery
from src.application.services.orchestrator.trading_cycle_entry import prepare_orchestrator_run_loop
from src.application.services.orchestrator.warm_up_buffer_guard import await_stream_warm_up_gate
from src.application.services.orchestrator.watchdog_service import start_ingestion_watchdog
from src.application.services.orchestrator.ws_bootstrap import (
    setup_trading_session,
    start_orchestrator_streams,
    subscribe_account_transactions,
)


async def setup_session(orch: Any) -> bool:
    """Autentica Deriv e prepara sessao de trading."""
    return await setup_trading_session(orch)


async def start_streams(orch: Any) -> bool:
    """Inicia streams OHLC e ticks do cluster."""
    return await start_orchestrator_streams(orch)


async def subscribe_transactions(orch: Any) -> None:
    """Assina transacoes de conta para reconciliacao de saldo."""
    await subscribe_account_transactions(orch)


async def stop_orchestrator(orch: Any) -> None:
    """Encerra conexoes de infraestrutura do orquestrador."""
    await close_infrastructure_connections(orch)


def emergency_save_session_state(orch: Any) -> None:
    """Persiste bundle financeiro de risco em session_state.json em modo de emergencia."""
    state_mgr = getattr(orch, "state_mgr", None)
    if state_mgr is None or type(state_mgr).__name__ != "StateManager":
        return
    state_mgr.state.current_balance = float(orch.state.balance)
    if state_mgr.state.initial_balance <= 0.0:
        state_mgr.state.initial_balance = float(orch.risk_manager.initial_bankroll)
    payload = {
        "initial_balance": state_mgr.state.initial_balance,
        "current_balance": state_mgr.state.current_balance,
        "daily_stop_win_target": state_mgr.state.daily_stop_win_target,
        "total_trades_today": state_mgr.state.total_trades_today,
        "stop_win_triggered": state_mgr.state.stop_win_triggered,
        "emergency_shutdown": True,
        "total_session_profit": orch.risk_manager.total_session_profit,
        "risk": orch.risk_manager.get_state(),
    }
    state_mgr.persistence.save(payload)


def _enforce_post_settlement_deadlock_exit(orch: Any) -> None:
    """Agenda reconciliacao passiva quando pos-liquidacao acumula incompletos; nao reinicia o loop."""
    streak = int(getattr(orch, "_post_settlement_incomplete_streak", 0))
    deadlock = bool(getattr(orch, "_post_settlement_deadlock", False))
    if not deadlock and streak < 2:
        return
    orch.logger.info(
        "SETTLE: incompleto pos-liquidacao (streak=%d); reconciliacao passiva via portfolio",
        streak,
    )
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run_passive_settlement_reconcile(orch))
    except RuntimeError:
        pass
    recover_post_settlement_loop_transparently(orch)


async def _run_passive_settlement_reconcile(orch: Any) -> None:
    """Consulta portfolio/Redis e limpa contratos ja liquidados sem ACK."""
    cleaner = SettlementOrphanCleaner(orch)
    cleared = await cleaner.passive_reconcile()
    if cleared:
        orch.logger.info("SETTLE: estado local alinhado ao broker; ciclo pode prosseguir")


def _recovery_pending_total(orch: Any) -> float:
    """Soma passivo pendente vivo quando o risk manager expoe a API."""
    rm = getattr(orch, "risk_manager", None)
    if rm is None:
        return 0.0
    pending_fn = getattr(rm, "pending_loss_total", None)
    if callable(pending_fn):
        return float(pending_fn())
    pending_map = getattr(rm, "pending_loss", {}) or {}
    if isinstance(pending_map, dict):
        return float(sum(float(v) for v in pending_map.values()))
    return 0.0


def align_exec_empty_recovery_signature_cooldown(orch: Any) -> float:
    """Apos EXEC_EMPTY em recovery, espera no maximo 45s (ou fronteira se menor)."""
    if bool(getattr(orch, "_last_cycle_cluster_executed", False)):
        return 0.0
    if _recovery_pending_total(orch) <= 0.0:
        return 0.0
    delay = float(seconds_until_next_signature_boundary(orch))
    orch_cfg = orch.config.get("orchestrator") if isinstance(getattr(orch, "config", None), dict) else {}
    if not isinstance(orch_cfg, dict):
        orch_cfg = {}
    try:
        empty_cap = float(orch_cfg.get("exec_empty_retry_seconds", 45))
    except (TypeError, ValueError):
        empty_cap = 45.0
    delay = min(max(delay, 1.0), max(15.0, empty_cap))
    orch._cooldown_until = time.time() + delay
    orch._post_settlement_incomplete_streak = 0
    return delay


async def run_orchestrator_main_loop(orch: Any) -> None:
    """Loop principal assincrono com reconexao, persistencia e ciclos M5."""
    if not await setup_session(orch):
        orch.logger.error("INIT: Abortando motor (falha em PAT, OTP ou WebSocket).")
        return
    if not await start_streams(orch):
        orch.logger.error("INIT: Abortando motor (falha ao sincronizar velas OHLC).")
        return
    prepare_orchestrator_run_loop(orch)
    await await_stream_warm_up_gate(orch)
    await start_settlement_worker(orch)
    await start_ingestion_watchdog(orch)
    await orch._run_trading_cycle_if_ready()
    align_exec_empty_recovery_signature_cooldown(orch)
    reconcile_counter = 0
    orch_cfg = orch.config.get("orchestrator") if isinstance(orch.config.get("orchestrator"), dict) else {}
    reconnect_delay = float(orch_cfg.get("ws_reconnect_delay_seconds", 8.0))
    while orch.running:
        cooldown = float(getattr(orch, "_cooldown_until", 0.0))
        if cooldown > 0.0 and time.time() < cooldown:
            await asyncio.sleep(max(0.1, cooldown - time.time()))
            continue
        _enforce_post_settlement_deadlock_exit(orch)
        await orch._tick_idle_cycle_watchdog()
        await orch._tick_interval_cycle_if_due()
        current_signature = get_data_state_signature(orch)
        if current_signature and current_signature == orch.last_data_signature and orch.ws.is_running:
            await asyncio.sleep(0.1)
            continue
        await asyncio.sleep(1)
        if not orch.ws.is_running:
            if await setup_session(orch) and await start_streams(orch):
                orch.logger.info("RECOV: WebSocket restaurado.")
                release_trading_cycle_after_reconnect(orch)
                await await_stream_warm_up_gate(orch)
                await reconcile_after_ws_recovery(orch)
                reconnect_delay = float(orch_cfg.get("ws_reconnect_delay_seconds", 8.0))
            else:
                orch.logger.warning("RECOV: broker indisponivel; nova tentativa em %.0fs.", reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, 60.0)
            continue
        reconcile_counter += 1
        await save_full_state(orch)
        if reconcile_counter >= orch.config["orchestrator"].get("reconcile_interval_seconds", 60):
            if orch.state.active_contracts:
                await orch.executor.reconcile()
            reconcile_counter = 0
        await orch._run_trading_cycle_if_ready()
        align_exec_empty_recovery_signature_cooldown(orch)
