"""Lida com fluxos de dados em tempo real e mantém histórico local para múltiplos símbolos."""

import asyncio
import logging
from datetime import datetime

import numpy as np

from src.domain.models.market_data import Candle
from src.infrastructure.api.deriv_granularity import normalize_granularity_seconds
from src.infrastructure.api.websocket_manager import WebSocketManager
from src.infrastructure.handlers.history_fetch import fetch_paginated_candle_history, parse_history_fetch_config
from src.infrastructure.handlers.stream_ohlc_fetch import fetch_candle_close_rows, fetch_candle_ohlc_rows
from src.infrastructure.handlers.tick_buffer import TickBuffer


_TYPE_VALUE_ERRORS = (TypeError, ValueError)
_KEY_TYPE_VALUE_ERRORS = (KeyError, TypeError, ValueError)


class StreamHandler:
    """Gerencia fluxos de dados de mercado em tempo real (OHLC) para múltiplos símbolos."""

    def __init__(self, ws_manager: WebSocketManager, symbols: list[str], data_config: dict):
        """Inicializa o manipulador."""
        self.ws = ws_manager
        self.symbols = symbols
        self.candles = {s: [] for s in symbols}
        self.config = data_config
        self.history_limit = self.config.get("buffer_limit", 1000)
        self.logger = logging.getLogger("AETH")
        requested = int(self.config.get("granularity", 60))
        self.granularity = normalize_granularity_seconds(requested)
        if self.granularity != requested:
            self.logger.warning(
                "DATA: granularity %ss nao suportada pela Deriv; usando %ss",
                requested,
                self.granularity,
            )
        self.candle_callback = None
        self.is_synchronized = False
        self.tick_buffer = TickBuffer(symbols)
        self._last_bar_epoch: dict[str, int | None] = dict.fromkeys(symbols)

    def _resolve_fetch_count(self) -> int:
        """Define quantas velas buscar na sincronizacao inicial."""
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
        """Ativa subscrições do cluster e busca histórico paralelo."""
        fetch_count = self._resolve_fetch_count()
        quiet = self._history_sync_quiet(fetch_count)
        self.is_synchronized = False

        if not self.ws.is_running:
            raise ConnectionError("STREAM: WebSocket desconectado antes da sincronização.")

        sync_log = self.logger.debug if quiet else self.logger.info
        sync_log(
            "DATA: Sincronizando historico | %d simbolos | alvo %d velas | aguarde",
            len(self.symbols),
            fetch_count,
        )
        fetch_cfg = parse_history_fetch_config(self.config)
        total = len(self.symbols)
        for index, symbol in enumerate(self.symbols, start=1):
            sync_log("DATA: Historico %s (%d/%d) | iniciando", symbol, index, total)
            await self._fetch_symbol_history(symbol, fetch_count, fetch_cfg=fetch_cfg, quiet=quiet)
            bars = len(self.candles.get(symbol, []))
            sync_log("DATA: Historico %s (%d/%d) | %d velas", symbol, index, total, bars)
            symbol_delay = float(fetch_cfg["symbol_delay"])
            if symbol_delay > 0:
                await asyncio.sleep(symbol_delay)
        if self.symbols:
            bars = len(self.candles.get(self.symbols[0], []))
            label = self.symbols[0] if len(self.symbols) == 1 else f"{len(self.symbols)} simbolos"
            self.logger.info("DATA: Buffer pronto | %s | %d velas", label, bars)
        if not self.ws.is_running:
            raise ConnectionError("STREAM: WebSocket desconectado após sincronização histórica.")

        self.ws.subscribe("ohlc", self._on_candle)
        self.ws.subscribe("tick", self._on_tick)
        self.candle_callback = callback

        self.logger.debug(f"STRM: Ativando fluxo de velas ({self.granularity}s) do Enxame...")
        sub_tasks = [
            self.ws.send(
                {
                    "ticks_history": s,
                    "style": "candles",
                    "granularity": self.granularity,
                    "subscribe": 1,
                    "end": "latest",
                    "count": 1,
                }
            )
            for s in self.symbols
        ]
        await asyncio.gather(*sub_tasks)
        tick_tasks = [
            self.ws.send(
                {
                    "ticks_history": s,
                    "style": "ticks",
                    "subscribe": 1,
                    "end": "latest",
                    "count": 1,
                }
            )
            for s in self.symbols
        ]
        await asyncio.gather(*tick_tasks)
        self.is_synchronized = True
        self.logger.debug("DATA: Sincronia concluída. Buffer histórico em conformidade.")

    async def _fetch_symbol_history(
        self,
        symbol: str,
        count: int,
        *,
        fetch_cfg: dict | None = None,
        quiet: bool = False,
    ) -> int:
        """Busca historico paginado para um simbolo e retorna quantidade armazenada."""
        cfg = fetch_cfg or parse_history_fetch_config(self.config)
        existing = self.candles.get(symbol, [])
        merged = await fetch_paginated_candle_history(
            self.ws,
            symbol=symbol,
            granularity=self.granularity,
            target=count,
            fetch_cfg=cfg,
            logger=self.logger,
            existing=existing,
            quiet=quiet,
        )
        self.candles[symbol] = merged
        self.logger.debug("DATA: Historico %s | %d velas (alvo %d)", symbol, len(merged), count)
        return len(merged)

    async def ensure_cluster_history(self, target: int) -> None:
        """Continua backfill sequencial ate atingir o alvo de velas por simbolo."""
        goal = max(1, int(target))
        fetch_cfg = parse_history_fetch_config(self.config)
        for symbol in self.symbols:
            before = len(self.candles.get(symbol, []))
            if before >= goal:
                continue
            after = await self._fetch_symbol_history(symbol, goal, fetch_cfg=fetch_cfg)
            if after > before:
                self.logger.info(
                    "DATA: Backfill %s | %d -> %d velas (alvo %d)",
                    symbol,
                    before,
                    after,
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
        if symbol not in self.candles:
            return

        candle = Candle(
            symbol=symbol,
            open=float(o["open"]),
            high=float(o["high"]),
            low=float(o["low"]),
            close=float(o["close"]),
            time=datetime.fromtimestamp(o["open_time"]),
            epoch=o["open_time"],
        )

        if self.candles[symbol] and self.candles[symbol][-1].epoch == candle.epoch:
            self.candles[symbol][-1] = candle
            self.tick_buffer.on_bar_update(symbol, candle.epoch)
        else:
            prev_epoch = self._last_bar_epoch.get(symbol)
            if prev_epoch is not None and prev_epoch != candle.epoch:
                self.tick_buffer.on_bar_close(symbol, prev_epoch)
            self.candles[symbol].append(candle)
            self._last_bar_epoch[symbol] = candle.epoch
            self.tick_buffer.on_bar_update(symbol, candle.epoch)

        if len(self.candles[symbol]) > self.history_limit:
            self.candles[symbol].pop(0)

        if self.candle_callback:
            await self.candle_callback(candle)

    async def _on_tick(self, data):
        """Processa ticks recebidos para microestrutura."""
        tick = data.get("tick")
        if not isinstance(tick, dict):
            return
        symbol = tick.get("symbol")
        if symbol not in self.tick_buffer.symbols:
            return
        epoch = tick.get("epoch")
        quote = tick.get("quote")
        if epoch is None or quote is None:
            return
        try:
            epoch_ms = int(float(epoch) * 1000)
            price = float(quote)
        except _TYPE_VALUE_ERRORS:
            return
        self.tick_buffer.record_tick(symbol, epoch_ms, price)

    def get_numpy_series(self, symbol: str, field: str = "close") -> np.ndarray:
        """Retorna uma série numpy de um campo específico para o timeframe M1."""
        history = self.candles.get(symbol, [])

        if not history:
            return np.array([])
        return np.array([getattr(c, field) for c in history])

    def get_last_candle_epoch(self, symbol: str) -> int | None:
        """Retorna o epoch da ultima vela M1 armazenada para o simbolo."""
        history = self.candles.get(symbol, [])
        if not history:
            return None
        return int(history[-1].epoch)

    async def fetch_candle_ohlc(
        self, symbol: str, granularity: int, count: int
    ) -> list[tuple[float, float, float, float]]:
        """Busca velas OHLC (open, high, low, close) por granularidade sem alterar o buffer local."""
        if symbol not in self.symbols:
            return []
        return await fetch_candle_ohlc_rows(self.ws, symbol, granularity, count, self.logger)

    async def fetch_candle_closes(self, symbol: str, granularity: int, count: int) -> list[float]:
        """Busca fechamentos OHLC para outra granularidade (ex. M5=300s) sem mudar o buffer local."""
        if symbol not in self.symbols:
            return []
        return await fetch_candle_close_rows(self.ws, symbol, granularity, count, self.logger)
