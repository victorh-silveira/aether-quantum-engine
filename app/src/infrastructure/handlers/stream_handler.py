"""Lida com fluxos de dados em tempo real e mantém histórico local para múltiplos símbolos."""

import asyncio
import logging
from typing import Any

import numpy as np

from src.domain.models.market_data import Candle
from src.infrastructure.api.websocket_manager import WebSocketManager
from src.infrastructure.handlers.history_fetch import fetch_paginated_candle_history, parse_history_fetch_config
from src.infrastructure.handlers.stream_candle_apply import apply_candle_update, candle_from_ohlc
from src.infrastructure.handlers.stream_ohlc_fetch import fetch_candle_close_rows, fetch_candle_ohlc_rows
from src.infrastructure.handlers.stream_reconnect import execute_stream_reconnect
from src.infrastructure.handlers.stream_tick_sidecar import handle_stream_tick, persist_closed_bar
from src.infrastructure.handlers.stream_timeframe import (
    ohlc_payload_granularity,
    resolve_dual_granularity,
    resolve_micro_fetch_count,
    subscribe_candle_streams,
    subscribe_tick_streams,
)
from src.infrastructure.handlers.tick_buffer import TickBuffer


class StreamHandler:
    """Gerencia fluxos macro M15 (DL) e micro M1 (execucao) para multiplos simbolos."""

    def __init__(self, ws_manager: WebSocketManager, symbols: list[str], data_config: dict, *, market_writer=None):
        """Inicializa o manipulador."""
        self.ws = ws_manager
        self.symbols = symbols
        self.config = data_config
        self._market_writer = market_writer
        self.history_limit = self.config.get("buffer_limit", 1000)
        self.logger = logging.getLogger("AETH")
        self.macro_granularity, self.micro_granularity = resolve_dual_granularity(data_config)
        self.granularity = self.macro_granularity
        self.macro_candles = {s: [] for s in symbols}
        self.micro_candles = {s: [] for s in symbols}
        self.candles = self.macro_candles
        self.candle_callback = None
        self.is_synchronized = False
        self.tick_buffer = TickBuffer(symbols)
        self._last_macro_bar_epoch: dict[str, int | None] = dict.fromkeys(symbols)
        self._last_micro_bar_epoch: dict[str, int | None] = dict.fromkeys(symbols)

    def _resolve_fetch_count(self) -> int:
        """Define quantas velas macro buscar na sincronizacao inicial."""
        startup = self.config.get("_startup_fetch_count")
        if startup is not None:
            return max(1, int(startup))
        if "fetch_count" in self.config:
            return max(1, int(self.config["fetch_count"]))
        history_bars = int(self.config.get("history_bars", 0))
        if history_bars > 0:
            warmup = int(self.config.get("history_warmup_bars", 32))
            return history_bars + warmup
        return 500

    def _history_sync_quiet(self, goal: int) -> bool:
        """Indica sync inicial curto (inferencia) com logs reduzidos."""
        if self.config.get("_startup_fetch_count") is not None:
            return True
        return int(goal) <= 512

    async def start_candle_stream(self, callback):
        """Ativa subscricoes macro/micro e busca historico paralelo."""
        macro_count = self._resolve_fetch_count()
        micro_count = resolve_micro_fetch_count(self.config)
        quiet = self._history_sync_quiet(macro_count)
        self.is_synchronized = False
        if not self.ws.is_running:
            raise ConnectionError("STREAM: WebSocket desconectado antes da sincronização.")
        sync_log = self.logger.debug if quiet else self.logger.info
        sync_log(
            "DATA: Sincronizando historico | %d simbolos | macro=%ds x%d | micro=%ds x%d",
            len(self.symbols),
            self.macro_granularity,
            macro_count,
            self.micro_granularity,
            micro_count,
        )
        fetch_cfg = parse_history_fetch_config(self.config)
        total = len(self.symbols)
        for index, symbol in enumerate(self.symbols, start=1):
            sync_log("DATA: Historico %s (%d/%d) | iniciando", symbol, index, total)
            await self._fetch_symbol_history(
                symbol, macro_count, granularity=self.macro_granularity, store=self.macro_candles, quiet=quiet
            )
            await self._fetch_symbol_history(
                symbol, micro_count, granularity=self.micro_granularity, store=self.micro_candles, quiet=quiet
            )
            bars = len(self.macro_candles.get(symbol, []))
            sync_log(
                "DATA: Historico %s (%d/%d) | macro=%d micro=%d",
                symbol,
                index,
                total,
                bars,
                len(self.micro_candles[symbol]),
            )
            symbol_delay = float(fetch_cfg["symbol_delay"])
            if symbol_delay > 0:
                await asyncio.sleep(symbol_delay)
        if self.symbols:
            bars = len(self.macro_candles.get(self.symbols[0], []))
            self.logger.info("DATA | buffer macro %d simbolos x %d velas M15", len(self.symbols), bars)
        if not self.ws.is_running:
            raise ConnectionError("STREAM: WebSocket desconectado após sincronização histórica.")
        self.ws.subscribe("ohlc", self._on_candle)
        self.ws.subscribe("tick", self._on_tick)
        self.candle_callback = callback
        self.logger.debug(
            "STRM: Ativando fluxos M15=%ss e M1=%ss para %d simbolos",
            self.macro_granularity,
            self.micro_granularity,
            len(self.symbols),
        )
        await subscribe_candle_streams(self.ws, self.symbols, self.macro_granularity)
        await subscribe_candle_streams(self.ws, self.symbols, self.micro_granularity)
        await subscribe_tick_streams(self.ws, self.symbols)
        self.is_synchronized = True
        self.logger.debug("DATA: Sincronia concluída. Buffer histórico em conformidade.")

    async def _fetch_symbol_history(
        self,
        symbol: str,
        count: int,
        *,
        granularity: int,
        store: dict[str, list],
        fetch_cfg: dict | None = None,
        quiet: bool = False,
    ) -> int:
        """Busca historico paginado para um simbolo e retorna quantidade armazenada."""
        cfg = fetch_cfg or parse_history_fetch_config(self.config)
        existing = store.get(symbol, [])
        merged = await fetch_paginated_candle_history(
            self.ws,
            symbol=symbol,
            granularity=granularity,
            target=count,
            fetch_cfg=cfg,
            logger=self.logger,
            existing=existing,
            quiet=quiet,
        )
        store[symbol] = merged
        self.logger.debug("DATA: Historico %s g=%ss | %d velas (alvo %d)", symbol, granularity, len(merged), count)
        return len(merged)

    async def ensure_cluster_history(self, target: int) -> None:
        """Continua backfill macro sequencial ate atingir o alvo de velas por simbolo."""
        goal = max(1, int(target))
        fetch_cfg = parse_history_fetch_config(self.config)
        for symbol in self.symbols:
            before = len(self.macro_candles.get(symbol, []))
            if before >= goal:
                continue
            after = await self._fetch_symbol_history(
                symbol, goal, granularity=self.macro_granularity, store=self.macro_candles, fetch_cfg=fetch_cfg
            )
            if after > before:
                self.logger.info("DATA: Backfill %s | %d -> %d velas macro (alvo %d)", symbol, before, after, goal)
            symbol_delay = float(fetch_cfg["symbol_delay"])
            if symbol_delay > 0:
                await asyncio.sleep(symbol_delay)

    async def _on_candle(self, data):
        """Processa atualizações de OHLC recebidas."""
        if "ohlc" not in data:
            return
        o = data["ohlc"]
        symbol = o["symbol"]
        gran = ohlc_payload_granularity(o, self.macro_granularity, self.micro_granularity)
        candle = candle_from_ohlc(symbol, o)
        if gran == self.macro_granularity:
            await self._apply_macro_candle(symbol, candle)
        elif gran == self.micro_granularity:
            await self._apply_micro_candle(symbol, candle)

    async def _apply_macro_candle(self, symbol: str, candle: Candle):
        """Atualiza buffer macro M15 e microestrutura por barra fechada."""
        if symbol not in self.macro_candles:
            return
        result = apply_candle_update(
            self.macro_candles,
            self._last_macro_bar_epoch,
            symbol,
            candle,
            limit=int(self.history_limit),
        )
        if result.event == "update":
            self.tick_buffer.on_bar_update(symbol, candle.epoch)
            return
        if result.closed_epoch is not None:
            history = self.macro_candles.get(symbol, [])
            closed = history[-2] if len(history) >= 2 else candle
            micro = self.tick_buffer.on_bar_close(symbol, result.closed_epoch)
            await persist_closed_bar(
                self, symbol, result.closed_epoch, closed, micro, granularity=self.macro_granularity
            )
        self.tick_buffer.on_bar_update(symbol, candle.epoch)

    async def _apply_micro_candle(self, symbol: str, candle: Candle):
        """Atualiza buffer micro M1 e dispara callback operacional."""
        if symbol not in self.micro_candles:
            return
        apply_candle_update(
            self.micro_candles,
            self._last_micro_bar_epoch,
            symbol,
            candle,
            limit=max(512, int(self.history_limit // 16)),
        )
        if self.candle_callback:
            await self.candle_callback(candle)

    async def _on_tick(self, data):
        """Processa ticks recebidos para microestrutura macro."""
        await handle_stream_tick(self, data)

    def get_numpy_series(self, symbol: str, field: str = "close") -> np.ndarray:
        """Retorna serie numpy macro M15 para deep learning."""
        return self._series_from_store(self.macro_candles, symbol, field)

    def get_micro_numpy_series(self, symbol: str, field: str = "close") -> np.ndarray:
        """Retorna serie numpy micro M1 para assinatura operacional."""
        return self._series_from_store(self.micro_candles, symbol, field)

    @staticmethod
    def _series_from_store(store: dict[str, list], symbol: str, field: str) -> np.ndarray:
        """Extrai campo numerico do buffer OHLC informado."""
        history = store.get(symbol, [])
        if not history:
            return np.array([])
        return np.array([getattr(c, field) for c in history])

    def get_last_candle_epoch(self, symbol: str) -> int | None:
        """Retorna epoch da ultima vela macro M15."""
        return self._last_epoch_from_store(self.macro_candles, symbol)

    def get_last_micro_candle_epoch(self, symbol: str) -> int | None:
        """Retorna epoch da ultima vela micro M1."""
        return self._last_epoch_from_store(self.micro_candles, symbol)

    @staticmethod
    def _last_epoch_from_store(store: dict[str, list], symbol: str) -> int | None:
        """Retorna epoch da ultima vela no buffer informado."""
        history = store.get(symbol, [])
        if not history:
            return None
        return int(history[-1].epoch)

    async def fetch_candle_ohlc(
        self, symbol: str, granularity: int, count: int
    ) -> list[tuple[float, float, float, float]]:
        """Busca velas OHLC por granularidade sem alterar o buffer local."""
        if symbol not in self.symbols:
            return []
        return await fetch_candle_ohlc_rows(self.ws, symbol, granularity, count, self.logger)

    async def fetch_candle_closes(self, symbol: str, granularity: int, count: int) -> list[float]:
        """Busca fechamentos OHLC por granularidade sem alterar o buffer local."""
        if symbol not in self.symbols:
            return []
        return await fetch_candle_close_rows(self.ws, symbol, granularity, count, self.logger)

    async def reconnect_stream(self, orch: Any) -> bool:
        """Reinicializa WebSocket e subscricoes de mercado apos inanicao de ticks."""
        return await execute_stream_reconnect(orch, self)
