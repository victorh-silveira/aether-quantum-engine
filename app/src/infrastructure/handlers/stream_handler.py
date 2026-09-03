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
from src.infrastructure.handlers.stream_sync_start import sync_triple_candle_history
from src.infrastructure.handlers.stream_tick_sidecar import handle_stream_tick, persist_closed_bar
from src.infrastructure.handlers.stream_timeframe import (
    ohlc_payload_granularity,
    resolve_triple_granularity,
)
from src.infrastructure.handlers.tick_buffer import TickBuffer


class StreamHandler:
    """Gerencia fluxos MACRO/MICRO/MINI e ticks para multiplos simbolos."""

    def __init__(self, ws_manager: WebSocketManager, symbols: list[str], data_config: dict, *, market_writer=None):
        """Inicializa o manipulador."""
        self.ws = ws_manager
        self.symbols = symbols
        self.config = data_config
        self._market_writer = market_writer
        self.history_limit = self.config.get("buffer_limit", 1000)
        self.logger = logging.getLogger("AETH")
        self.macro_granularity, self.micro_granularity, self.mini_granularity = resolve_triple_granularity(data_config)
        self.granularity = self.macro_granularity
        self.macro_candles = {s: [] for s in symbols}
        self.micro_candles = {s: [] for s in symbols}
        self.mini_candles = {s: [] for s in symbols}
        self.candles = self.macro_candles
        self.candle_callback = None
        self.is_synchronized = False
        self.tick_buffer = TickBuffer(symbols)
        self._last_macro_bar_epoch: dict[str, int | None] = dict.fromkeys(symbols)
        self._last_micro_bar_epoch: dict[str, int | None] = dict.fromkeys(symbols)
        self._last_mini_bar_epoch: dict[str, int | None] = dict.fromkeys(symbols)
        self._reconnect_in_progress = False

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
        """Indica sync curto de inferencia com logs reduzidos (treino permanece verboso)."""
        if self.config.get("_startup_quiet") is True:
            return True
        if self.config.get("_startup_fetch_count") is not None:
            return False
        return int(goal) <= 512

    async def start_candle_stream(self, callback):
        """Ativa subscricoes MACRO/MICRO/MINI e busca historico paralelo."""
        await sync_triple_candle_history(self, callback)

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

    async def ensure_cluster_history(self, target: int, *, timeframe: str = "macro") -> None:
        """Continua backfill do buffer de treino ate atingir o alvo de velas por simbolo."""
        goal = max(1, int(target))
        fetch_cfg = parse_history_fetch_config(self.config)
        tf = str(timeframe).strip().lower()
        if tf == "micro":
            gran = self.micro_granularity
            store = self.micro_candles
            label = "micro"
        elif tf == "mini":
            gran = self.mini_granularity
            store = self.mini_candles
            label = "mini"
        else:
            gran = self.macro_granularity
            store = self.macro_candles
            label = "macro"
        for symbol in self.symbols:
            before = len(store.get(symbol, []))
            if before >= goal:
                continue
            after = await self._fetch_symbol_history(symbol, goal, granularity=gran, store=store, fetch_cfg=fetch_cfg)
            if after > before:
                self.logger.info(
                    "DATA: Backfill %s | %d -> %d velas %s (alvo %d)",
                    symbol,
                    before,
                    after,
                    label,
                    goal,
                )
            symbol_delay = float(fetch_cfg["symbol_delay"])
            if symbol_delay > 0:
                await asyncio.sleep(symbol_delay)

    async def _on_candle(self, data):
        """Processa atualizações de OHLC recebidas."""
        if "ohlc" not in data:
            return
        o = data["ohlc"]
        symbol = o["symbol"]
        gran = ohlc_payload_granularity(o, self.macro_granularity, self.micro_granularity, self.mini_granularity)
        candle = candle_from_ohlc(symbol, o)
        if gran == self.macro_granularity:
            await self._apply_macro_candle(symbol, candle)
        if gran == self.micro_granularity:
            await self._apply_micro_candle(symbol, candle)
        if (
            gran == self.mini_granularity
            and self.mini_granularity != self.micro_granularity
            or gran == self.micro_granularity
            and self.mini_granularity == self.micro_granularity
        ):
            await self._apply_mini_candle(symbol, candle)

    async def _apply_macro_candle(self, symbol: str, candle: Candle):
        """Atualiza buffer macro e microestrutura por barra fechada."""
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
        """Atualiza buffer micro e dispara callback operacional."""
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

    async def _apply_mini_candle(self, symbol: str, candle: Candle):
        """Atualiza buffer MINI (tape curto)."""
        if symbol not in self.mini_candles:
            return
        apply_candle_update(
            self.mini_candles,
            self._last_mini_bar_epoch,
            symbol,
            candle,
            limit=max(512, int(self.history_limit // 8)),
        )

    async def _on_tick(self, data):
        """Processa ticks recebidos para microestrutura macro."""
        await handle_stream_tick(self, data)

    def get_numpy_series(self, symbol: str, field: str = "close") -> np.ndarray:
        """Retorna serie numpy macro para deep learning."""
        return self._series_from_store(self.macro_candles, symbol, field)

    def get_micro_numpy_series(self, symbol: str, field: str = "close") -> np.ndarray:
        """Retorna serie numpy micro para assinatura operacional."""
        return self._series_from_store(self.micro_candles, symbol, field)

    def get_mini_numpy_series(self, symbol: str, field: str = "close") -> np.ndarray:
        """Retorna serie numpy MINI para visao de tape curto."""
        series = self._series_from_store(self.mini_candles, symbol, field)
        if len(series) == 0:
            return self._series_from_store(self.micro_candles, symbol, field)
        return series

    @staticmethod
    def _series_from_store(store: dict[str, list], symbol: str, field: str) -> np.ndarray:
        """Extrai campo numerico do buffer OHLC informado."""
        history = store.get(symbol, [])
        if not history:
            return np.array([])
        return np.array([getattr(c, field) for c in history])

    def get_last_candle_epoch(self, symbol: str) -> int | None:
        """Retorna epoch da ultima vela macro."""
        return self._last_epoch_from_store(self.macro_candles, symbol)

    def get_last_micro_candle_epoch(self, symbol: str) -> int | None:
        """Retorna epoch da ultima vela micro."""
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
        if self._reconnect_in_progress:
            self.logger.debug("STREAM: Reconnect ja em andamento. Retornando silenciosamente.")
            return False
        self._reconnect_in_progress = True
        try:
            return await execute_stream_reconnect(orch, self)
        finally:
            self._reconnect_in_progress = False
