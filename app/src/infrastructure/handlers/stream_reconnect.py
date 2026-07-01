"""Reconexao controlada de subscricoes OHLC/tick do StreamHandler."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from src.application.services.orchestrator.ws_bootstrap import subscribe_account_transactions, ws_connect_options


if TYPE_CHECKING:
    from src.infrastructure.handlers.stream_handler import StreamHandler


logger = logging.getLogger("AETH")


async def _resubscribe_market_channels(stream: StreamHandler) -> None:
    """Reenvia subscribe de velas e ticks para todos os simbolos."""
    sub_tasks = [
        stream.ws.send(
            {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": stream.granularity,
                "subscribe": 1,
                "end": "latest",
                "count": 1,
            }
        )
        for symbol in stream.symbols
    ]
    await asyncio.gather(*sub_tasks)
    tick_tasks = [
        stream.ws.send(
            {
                "ticks_history": symbol,
                "style": "ticks",
                "subscribe": 1,
                "end": "latest",
                "count": 1,
            }
        )
        for symbol in stream.symbols
    ]
    await asyncio.gather(*tick_tasks)


async def execute_stream_reconnect(orch: Any, stream: StreamHandler) -> bool:
    """Fecha WS, reabre sessao OTP e reativa fluxo de mercado sem backfill pesado."""
    callback = stream.candle_callback
    if callback is None:
        return False
    opts = ws_connect_options(orch)
    try:
        if orch.ws.ws:
            await orch.ws.close()
        session = await orch.auth.open_trading_session()
        await orch.ws.connect(session.ws_url, **opts)
        orch.ws.subscribe("proposal_open_contract", orch._on_contract_update)
        stream.ws.subscribe("ohlc", stream._on_candle)
        stream.ws.subscribe("tick", stream._on_tick)
        await subscribe_account_transactions(orch)
        await _resubscribe_market_channels(stream)
        stream.is_synchronized = True
        stream.tick_buffer.touch_activity()
        loop = asyncio.get_running_loop()
        orch._stream_ready_mono = loop.time()
        logger.info("WATCHDOG: stream de mercado reconectado")
        return True
    except Exception as exc:
        logger.warning("WATCHDOG: reconnect_stream falhou: %s", exc)
        return False
