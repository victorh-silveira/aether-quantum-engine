"""Reconexao controlada de subscricoes OHLC/tick do StreamHandler."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from src.application.services.orchestrator.warm_up_buffer_guard import schedule_stream_warm_up_barrier
from src.application.services.orchestrator.ws_bootstrap import subscribe_account_transactions, ws_connect_options
from src.infrastructure.handlers.stream_timeframe import subscribe_candle_streams, subscribe_tick_streams


if TYPE_CHECKING:
    from src.infrastructure.handlers.stream_handler import StreamHandler


logger = logging.getLogger("AETH")


def _continuous_stream_active(orch: Any) -> bool:
    """Indica sessao continua com stream OHLC ja sincronizado."""
    ready_mono = float(getattr(orch, "_stream_ready_mono", 0.0))
    return ready_mono > 0.0 and bool(getattr(orch, "running", False))


def _needs_profit_table_audit(orch: Any) -> bool:
    """True quando ha contratos ou reconciliacao pendente apos queda de rede."""
    state = getattr(orch, "state", None)
    if state is not None and getattr(state, "active_contracts", None) and state.active_contracts:
        return True
    risk = getattr(orch, "risk_manager", None)
    if risk is not None and getattr(risk, "active_contract_ids", None) and risk.active_contract_ids:
        return True
    return bool(getattr(orch, "_reconciliation_pending", False))


def _schedule_profit_audit(orch: Any, *, reason: str) -> None:
    """Delega auditoria profit_table ao TradeHandler quando disponivel."""
    trade_handler = getattr(orch, "trade_handler", None)
    schedule = getattr(trade_handler, "schedule_profit_table_audit", None)
    if callable(schedule):
        schedule(orch, reason=reason)


async def _resubscribe_market_channels(stream: StreamHandler) -> None:
    """Reenvia subscribe de velas macro/micro e ticks para todos os simbolos."""
    await subscribe_candle_streams(stream.ws, stream.symbols, stream.macro_granularity)
    await subscribe_candle_streams(stream.ws, stream.symbols, stream.micro_granularity)
    await subscribe_tick_streams(stream.ws, stream.symbols)


async def _fetch_fresh_otp_ws_url(orch: Any) -> str:
    """Renova OTP REST de uso unico antes de reabrir o WebSocket."""
    refresh = getattr(orch.auth, "refresh_otp_ws_url", None)
    if callable(refresh):
        return str(await refresh())
    session = await orch.auth.open_trading_session()
    return str(session.ws_url)


async def execute_stream_reconnect(orch: Any, stream: StreamHandler) -> bool:
    """Fecha WS, reabre sessao OTP e reativa fluxo de mercado sem backfill pesado."""
    callback = stream.candle_callback
    if callback is None:
        return False
    skip_ohlc_resub = _continuous_stream_active(orch)
    opts = ws_connect_options(orch)
    try:
        if orch.ws.ws:
            await orch.ws.close()
        ws_url = await _fetch_fresh_otp_ws_url(orch)
        await orch.ws.connect(ws_url, **opts)
        orch.ws.subscribe("proposal_open_contract", orch._on_contract_update)
        stream.ws.subscribe("ohlc", stream._on_candle)
        stream.ws.subscribe("tick", stream._on_tick)
        await subscribe_account_transactions(orch)
        if skip_ohlc_resub:
            logger.debug("WATCHDOG: sessao continua ativa; pulando resubscribe OHLC duplicado")
        else:
            await _resubscribe_market_channels(stream)
        stream.is_synchronized = True
        stream.tick_buffer.reset_live_accumulators()
        stream.tick_buffer.touch_activity()
        loop = asyncio.get_running_loop()
        orch._stream_ready_mono = loop.time()
        schedule_stream_warm_up_barrier(orch)
        logger.info("WATCHDOG: stream de mercado reconectado")
        if _needs_profit_table_audit(orch):
            _schedule_profit_audit(orch, reason="stream_reconnect")
        return True
    except Exception as exc:
        logger.warning("WATCHDOG: reconnect_stream falhou: %s", exc)
        _schedule_profit_audit(orch, reason="broker_unavailable")
        return False
