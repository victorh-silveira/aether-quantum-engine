"""Ciclo de vida, persistencia e sessao do Orquestrador."""

from typing import Any

from src.application.services.deep_learning.dl_deferred_train import cancel_deferred_symbol_training
from src.application.services.orchestrator.ws_bootstrap import (
    setup_trading_session,
    start_orchestrator_streams,
    subscribe_account_transactions,
)
from src.domain.risk.stop_win_target import resolve_stop_win_target


async def setup_session(orch: Any) -> bool:
    """Estabelece sessao de trading com autenticacao e WebSocket."""
    return await setup_trading_session(orch)


async def start_streams(orch: Any) -> bool:
    """Inicia streams OHLC e sincroniza velas historicas."""
    return await start_orchestrator_streams(orch)


async def subscribe_transactions(orch: Any) -> None:
    """Inscreve callback de transacoes de conta na Deriv."""
    await subscribe_account_transactions(orch)


def maybe_reset_daily_risk_session(orch: Any, epoch: int) -> None:
    """Reinicia stop win no inicio de cada dia UTC (vela ancora)."""
    day_key = int(epoch) // 86400
    if orch._risk_session_day_key == day_key:
        return
    orch._risk_session_day_key = day_key
    bal = float(orch.state.balance)

    risk_cfg = orch.config.get("risk_management", {})
    target = resolve_stop_win_target(risk_cfg, bal)

    orch.state_mgr.reset_daily_metrics(bal, target, day_key)
    orch.risk_manager.reset_daily_session(bal)
    orch.logger.info("RISK: Sessao diaria | banca=$%.2f | stop win diario ativo", bal)


async def save_full_state(orch: Any) -> None:
    """Persiste o snapshot completo do orquestrador (Estado + Risco + PnL)."""
    s = await orch.state.get_state()
    s.update(
        {
            "total_session_profit": orch.risk_manager.total_session_profit,
            "risk": orch.risk_manager.get_state(),
        }
    )
    orch.persistence.save(s)


async def stop_orchestrator(orch: Any) -> None:
    """Para o loop e fecha o WebSocket."""
    orch.running = False
    task = orch._post_settlement_task
    if task is not None and not task.done():
        task.cancel()
    cancel_deferred_symbol_training(orch)
    await orch.ws.close()
    orch.logger.debug("STOP: encerrado.")


def get_data_state_signature(orch: Any) -> str:
    """Calcula uma assinatura unica do estado dos dados de mercado."""
    sigs = []
    for sym in orch.symbols:
        if hasattr(orch, "stream") and hasattr(orch.stream, "candles"):
            history = orch.stream.candles.get(sym, [])
            if history:
                last_c = history[-1]  # pragma: no cover
                sigs.append(f"{sym}:{last_c.epoch}:{last_c.close}:{last_c.high}:{last_c.low}")  # pragma: no cover
    return "|".join(sigs)
