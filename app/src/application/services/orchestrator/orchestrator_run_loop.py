"""Ciclo de vida, persistencia e sessao do Orquestrador."""

import asyncio
from typing import Any

from src.application.services.orchestrator.graceful_shutdown import close_infrastructure_connections
from src.application.services.orchestrator.orchestrator_settlement_queue import start_settlement_worker
from src.application.services.orchestrator.orchestrator_state_restore import (
    persist_session_hash,
    session_hash_payload,
    sync_market_signature,
)
from src.application.services.orchestrator.session_target_bootstrap import (
    current_dlambert_redis_payload,
    current_session_redis_payload,
)
from src.application.services.orchestrator.settlement_reconciliation import reconcile_after_ws_recovery
from src.application.services.orchestrator.trading_cycle_entry import prepare_orchestrator_run_loop
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


async def save_full_state(orch: Any) -> None:
    """Persiste snapshot completo, sessao e assinaturas de mercado."""
    s = await orch.state.get_state()
    s.update(
        {
            "total_session_profit": orch.risk_manager.total_session_profit,
            "risk": orch.risk_manager.get_state(),
        }
    )
    sig = orch.get_data_state_signature()
    session = session_hash_payload(orch)
    start_bal, target_win = current_session_redis_payload(orch)
    dlambert_unit, linear_losses = current_dlambert_redis_payload(orch)
    save_bundle = getattr(orch.state_store, "save_state_bundle", None)
    skip_counter = int(getattr(orch, "_recovery_skip_counter", 0))
    if callable(save_bundle):
        await save_bundle(
            snapshot=s,
            session=session,
            market_sig=sig or None,
            recovery_skip_counter=skip_counter,
            session_start_balance=start_bal,
            session_target_win=target_win,
            dlambert_unit=dlambert_unit,
            consecutive_losses_linear=linear_losses,
        )
    else:
        await orch.state_store.save_snapshot(s)
        await persist_session_hash(orch)
        if sig:
            await sync_market_signature(orch, sig)
    if hasattr(orch.persistence, "save"):
        orch.persistence.save(s)


async def stop_orchestrator(orch: Any) -> None:
    """Encerra conexoes de infraestrutura do orquestrador."""
    await close_infrastructure_connections(orch)


def get_data_state_signature(orch: Any) -> str:
    """Monta assinatura combinada M1 (micro) e M15 (macro) por simbolo."""
    micro_parts: list[str] = []
    macro_parts: list[str] = []
    stream = getattr(orch, "stream", None)
    if stream is None:
        return ""
    for sym in orch.symbols:
        micro_hist = getattr(stream, "micro_candles", {}).get(sym, [])
        if micro_hist:
            last_m = micro_hist[-1]  # pragma: no cover
            micro_parts.append(
                f"{sym}:{last_m.epoch}:{last_m.open}:{last_m.close}:{last_m.high}:{last_m.low}"
            )  # pragma: no cover
        macro_hist = getattr(stream, "macro_candles", stream.candles).get(sym, [])
        if macro_hist:
            last_macro = macro_hist[-1]  # pragma: no cover
            macro_parts.append(
                f"{sym}:{last_macro.epoch}:{last_macro.open}:{last_macro.close}:{last_macro.high}:{last_macro.low}"
            )  # pragma: no cover
    micro_sig = "|".join(micro_parts)
    macro_sig = "|".join(macro_parts)
    if not micro_sig and not macro_sig:
        return ""
    return f"m1:{micro_sig};m15:{macro_sig}"


async def run_orchestrator_main_loop(orch: Any) -> None:
    """Loop principal assincrono com reconexao, persistencia e ciclos M1."""
    if not await setup_session(orch):
        orch.logger.error("INIT: Abortando motor (falha em PAT, OTP ou WebSocket).")
        return
    if not await start_streams(orch):
        orch.logger.error("INIT: Abortando motor (falha ao sincronizar velas OHLC).")
        return
    prepare_orchestrator_run_loop(orch)
    await start_settlement_worker(orch)
    await start_ingestion_watchdog(orch)
    await orch._run_trading_cycle_if_ready()
    reconcile_counter = 0
    orch_cfg = orch.config.get("orchestrator") if isinstance(orch.config.get("orchestrator"), dict) else {}
    reconnect_delay = float(orch_cfg.get("ws_reconnect_delay_seconds", 8.0))
    while orch.running:
        await orch._tick_idle_cycle_watchdog()
        await orch._tick_interval_cycle_if_due()
        current_signature = get_data_state_signature(orch)
        if current_signature and current_signature == orch.last_data_signature and orch.ws.is_running:
            await asyncio.sleep(0.1)  # pragma: no cover
            continue  # pragma: no cover
        await asyncio.sleep(1)
        if not orch.ws.is_running:
            if await setup_session(orch) and await start_streams(orch):
                orch.logger.info("RECOV: WebSocket restaurado.")
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
