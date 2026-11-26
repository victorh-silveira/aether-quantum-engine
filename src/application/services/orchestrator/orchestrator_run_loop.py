"""Ciclo principal assincrono, reconexao e persistencia do Orquestrador Laya."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from src.application.services.orchestrator.graceful_shutdown import close_infrastructure_connections
from src.application.services.orchestrator.orchestrator_data_signature import get_data_state_signature
from src.application.services.orchestrator.orchestrator_settlement_queue import (
    SettlementOrphanCleaner,
    start_settlement_worker,
)
from src.application.services.orchestrator.post_settlement_resilience import (
    recover_post_settlement_loop_transparently,
)
from src.application.services.orchestrator.trading_cycle_entry import prepare_orchestrator_run_loop
from src.application.services.orchestrator.trading_cycle_entry_guards import check_and_clear_boundary_cooldown
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
    """Inicia streams OHLC M5 no R_10."""
    return await start_orchestrator_streams(orch)


async def subscribe_transactions(orch: Any) -> None:
    """Assina transacoes da conta para reconciliacao de saldo."""
    await subscribe_account_transactions(orch)


async def stop_orchestrator(orch: Any) -> None:
    """Encerra conexoes de infraestrutura do orquestrador."""
    await close_infrastructure_connections(orch)


async def _run_passive_settlement_reconcile(orch: Any) -> None:
    """Consulta portfolio/Redis e limpa contratos ja liquidados sem ACK."""
    cleaner = SettlementOrphanCleaner(orch)
    cleared = await cleaner.passive_reconcile()
    if cleared:
        orch.logger.info("SETTLE.settle_align: estado local alinhado ao broker; ciclo pode prosseguir")


def _enforce_post_settlement_deadlock_exit(orch: Any) -> None:
    """Agenda reconciliacao passiva quando pos-liquidacao acumula incompletos."""
    streak = int(getattr(orch, "_post_settlement_incomplete_streak", 0))
    deadlock = bool(getattr(orch, "_post_settlement_deadlock", False))
    if not deadlock and streak < 2:
        return
    with contextlib.suppress(RuntimeError):
        asyncio.create_task(_run_passive_settlement_reconcile(orch))
    recover_post_settlement_loop_transparently(orch)


async def run_orchestrator_main_loop(orch: Any) -> None:
    """Loop principal assincrono continuo no R_10 M5."""
    ms = getattr(orch, "metrics_server", None)
    if ms is not None and hasattr(ms, "start"):
        with contextlib.suppress(Exception):
            start_res = ms.start()
            if asyncio.iscoroutine(start_res):
                await start_res
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
    while orch.running:
        cooldown = float(getattr(orch, "_cooldown_until", 0.0) or 0.0)
        if cooldown > 0.0 and time.time() < cooldown and not check_and_clear_boundary_cooldown(orch):
            await asyncio.sleep(min(1.0, max(0.05, cooldown - time.time())))
            continue
        _enforce_post_settlement_deadlock_exit(orch)
        await orch._tick_idle_cycle_watchdog()
        await orch._tick_interval_cycle_if_due()
        current_signature = get_data_state_signature(orch)
        if current_signature and current_signature == orch.last_data_signature and orch.ws.is_running:
            await asyncio.sleep(0.1)
            continue
        await asyncio.sleep(1.0)
