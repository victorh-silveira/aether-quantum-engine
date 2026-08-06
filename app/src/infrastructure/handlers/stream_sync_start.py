"""Sincronizacao inicial MACRO/MICRO/MINI do StreamHandler."""

from __future__ import annotations

import asyncio
from typing import Any

from src.infrastructure.handlers.history_fetch import parse_history_fetch_config
from src.infrastructure.handlers.stream_timeframe import (
    granularity_label,
    resolve_micro_fetch_count,
    resolve_mini_fetch_count,
    subscribe_candle_streams,
    subscribe_tick_streams,
)


def _resolve_sync_targets(handler: Any) -> tuple[int, int, int]:
    """Define quantas velas buscar; treino lean foca no micro (TCN)."""
    lean = bool(handler.config.get("_startup_train_lean"))
    startup = handler.config.get("_startup_fetch_count")
    micro_default = resolve_micro_fetch_count(handler.config)
    micro_count = max(1, int(startup)) if startup is not None else micro_default
    if lean:
        macro_count = min(128, micro_count)
        return macro_count, micro_count, 0
    macro_count = handler._resolve_fetch_count()
    mini_count = resolve_mini_fetch_count(handler.config)
    return macro_count, micro_count, mini_count


async def sync_triple_candle_history(handler: Any, callback) -> None:
    """Busca historico triplo, assina fluxos e marca sincronia no handler."""
    macro_count, micro_count, mini_count = _resolve_sync_targets(handler)
    quiet = handler._history_sync_quiet(max(macro_count, micro_count, mini_count, 1))
    handler.is_synchronized = False
    if not handler.ws.is_running:
        raise ConnectionError("STREAM: WebSocket desconectado antes da sincronização.")
    sync_log = handler.logger.debug if quiet else handler.logger.info
    sync_log(
        "DATA: Sincronizando historico | %d simbolos | macro=%ds x%d | micro=%ds x%d | mini=%ds x%d%s",
        len(handler.symbols),
        handler.macro_granularity,
        macro_count,
        handler.micro_granularity,
        micro_count,
        handler.mini_granularity,
        mini_count,
        " | lean_treino" if bool(handler.config.get("_startup_train_lean")) else "",
    )
    fetch_cfg = parse_history_fetch_config(handler.config)
    total = len(handler.symbols)
    for index, symbol in enumerate(handler.symbols, start=1):
        sync_log("DATA: Historico %s (%d/%d) | iniciando", symbol, index, total)
        if macro_count > 0:
            await handler._fetch_symbol_history(
                symbol, macro_count, granularity=handler.macro_granularity, store=handler.macro_candles, quiet=quiet
            )
        if micro_count > 0:
            await handler._fetch_symbol_history(
                symbol, micro_count, granularity=handler.micro_granularity, store=handler.micro_candles, quiet=quiet
            )
        if mini_count > 0:
            await handler._fetch_symbol_history(
                symbol, mini_count, granularity=handler.mini_granularity, store=handler.mini_candles, quiet=quiet
            )
        sync_log(
            "DATA: Historico %s (%d/%d) | macro=%d micro=%d mini=%d",
            symbol,
            index,
            total,
            len(handler.macro_candles[symbol]),
            len(handler.micro_candles[symbol]),
            len(handler.mini_candles[symbol]),
        )
        symbol_delay = float(fetch_cfg["symbol_delay"])
        if symbol_delay > 0:
            await asyncio.sleep(symbol_delay)
    if handler.symbols:
        sym0 = handler.symbols[0]
        handler.logger.info(
            "DATA | buffer %d simbolos | macro=%s x%d | micro=%s x%d | mini=%s x%d",
            len(handler.symbols),
            granularity_label(handler.macro_granularity),
            len(handler.macro_candles.get(sym0, [])),
            granularity_label(handler.micro_granularity),
            len(handler.micro_candles.get(sym0, [])),
            granularity_label(handler.mini_granularity),
            len(handler.mini_candles.get(sym0, [])),
        )
    if not handler.ws.is_running:
        raise ConnectionError("STREAM: WebSocket desconectado após sincronização histórica.")
    handler.ws.subscribe("ohlc", handler._on_candle)
    handler.ws.subscribe("tick", handler._on_tick)
    handler.candle_callback = callback
    await subscribe_candle_streams(handler.ws, handler.symbols, handler.macro_granularity)
    await subscribe_candle_streams(handler.ws, handler.symbols, handler.micro_granularity)
    if mini_count > 0 or not bool(handler.config.get("_startup_train_lean")):
        await subscribe_candle_streams(handler.ws, handler.symbols, handler.mini_granularity)
    await subscribe_tick_streams(handler.ws, handler.symbols)
    handler.is_synchronized = True
    handler.logger.debug("DATA: Sincronia concluída. Buffer histórico em conformidade.")
