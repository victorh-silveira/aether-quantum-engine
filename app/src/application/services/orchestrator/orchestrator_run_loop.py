"""Ciclo de vida, persistencia e sessao do Orquestrador."""

from typing import Any

from src.application.services.orchestrator.graceful_shutdown import close_infrastructure_connections
from src.application.services.orchestrator.orchestrator_state_restore import (
    persist_session_hash,
    session_hash_payload,
    sync_market_signature,
)
from src.application.services.orchestrator.session_target_bootstrap import current_session_redis_payload
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
    """Monta assinatura OHLC do ultimo candle fechado por simbolo."""
    sigs = []
    for sym in orch.symbols:
        if hasattr(orch, "stream") and hasattr(orch.stream, "candles"):
            history = orch.stream.candles.get(sym, [])
            if history:
                last_c = history[-1]  # pragma: no cover
                sigs.append(f"{sym}:{last_c.epoch}:{last_c.close}:{last_c.high}:{last_c.low}")  # pragma: no cover
    return "|".join(sigs)
