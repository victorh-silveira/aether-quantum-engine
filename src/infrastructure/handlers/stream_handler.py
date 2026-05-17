"""Lida com fluxos de dados em tempo real e mantém histórico local para múltiplos símbolos."""

import asyncio
import logging
from datetime import datetime

import numpy as np

from src.domain.models.market_data import Candle
from src.infrastructure.api.websocket_manager import WebSocketManager


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
        self.granularity = self.config.get("granularity", 60)
        self.logger = logging.getLogger("AETH")
        self.candle_callback = None
        self.is_synchronized = False

    async def start_candle_stream(self, callback):
        """Ativa subscrições do cluster e busca histórico paralelo."""
        fetch_count = self.config.get("fetch_count", 500)
        self.is_synchronized = False

        if not self.ws.is_running:
            raise ConnectionError("STREAM: WebSocket desconectado antes da sincronização.")

        self.logger.debug(f"DATA: Sincronizando Enxame Aegis ({len(self.symbols)} pares - OHLC {self.granularity}s)...")
        tasks = [self._fetch_symbol_history(s, fetch_count) for s in self.symbols]
        await asyncio.gather(*tasks)
        if not self.ws.is_running:
            raise ConnectionError("STREAM: WebSocket desconectado após sincronização histórica.")

        self.ws.subscribe("ohlc", self._on_candle)
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
        self.is_synchronized = True
        self.logger.debug("DATA: Sincronia concluída. Buffer histórico em conformidade.")

    async def _fetch_symbol_history(self, symbol: str, count: int):
        """Trabalhador interno para busca paralela de histórico."""
        request = {
            "ticks_history": symbol,
            "end": "latest",
            "style": "candles",
            "granularity": self.granularity,
            "count": count,
        }
        res = await self.ws.send(request)
        history = res.get("candles", [])
        self.candles[symbol] = [
            Candle(
                symbol=symbol,
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                time=datetime.fromtimestamp(c["epoch"]),
                epoch=c["epoch"],
            )
            for c in history
        ]

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
        else:
            self.candles[symbol].append(candle)

        if len(self.candles[symbol]) > self.history_limit:
            self.candles[symbol].pop(0)

        if self.candle_callback:
            await self.candle_callback(candle)

    def get_numpy_series(self, symbol: str, field: str = "close") -> np.ndarray:
        """Retorna uma série numpy de um campo específico para o timeframe M1."""
        history = self.candles.get(symbol, [])

        if not history:
            return np.array([])
        return np.array([getattr(c, field) for c in history])

    async def fetch_ticks_history(self, symbol: str, count: int) -> list[tuple[float, float]]:
        """Busca ultimos ticks via ticks_history (preco e epoch)."""
        if count <= 0 or not self.ws.is_running:
            return []
        req = {"ticks_history": symbol, "adjust_start_time": 1, "count": count, "end": "latest"}
        try:
            res = await self.ws.send(req)
        except Exception as e:
            self.logger.debug("DATA: ticks_history excecao %s: %s", symbol, e)
            return []
        if res.get("error"):
            return []
        hist = res.get("history") or {}
        prices = hist.get("prices") or []
        times = hist.get("times") or []
        out: list[tuple[float, float]] = []
        for i, p in enumerate(prices):
            try:
                ts = float(times[i]) if i < len(times) else float(i)
                out.append((ts, float(p)))
            except _TYPE_VALUE_ERRORS:
                continue
        return out

    async def fetch_candle_ohlc(
        self, symbol: str, granularity: int, count: int
    ) -> list[tuple[float, float, float, float]]:
        """Busca velas OHLC (open, high, low, close) por granularidade sem alterar o buffer local."""
        if count <= 0 or not self.ws.is_running:
            return []
        if symbol not in self.symbols:
            return []
        req = {
            "ticks_history": symbol,
            "end": "latest",
            "style": "candles",
            "granularity": granularity,
            "count": count,
        }
        try:
            res = await self.ws.send(req)
        except Exception as e:
            self.logger.debug("DATA: OHLC completo %s g=%s: %s", symbol, granularity, e)
            return []
        if res.get("error"):
            return []
        history = res.get("candles") or []
        out: list[tuple[float, float, float, float]] = []
        for c in history:
            try:
                out.append((float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])))
            except _KEY_TYPE_VALUE_ERRORS:
                continue
        return out

    async def fetch_candle_closes(self, symbol: str, granularity: int, count: int) -> list[float]:
        """Busca fechamentos OHLC para outra granularidade (ex. M5=300s) sem mudar o buffer local."""
        if count <= 0 or not self.ws.is_running:
            return []
        if symbol not in self.symbols:
            return []
        req = {
            "ticks_history": symbol,
            "end": "latest",
            "style": "candles",
            "granularity": granularity,
            "count": count,
        }
        try:
            res = await self.ws.send(req)
        except Exception as e:
            self.logger.debug("DATA: OHLC historia extra %s g=%s: %s", symbol, granularity, e)
            return []
        if res.get("error"):
            return []
        history = res.get("candles") or []
        out: list[float] = []
        for c in history:
            try:
                out.append(float(c["close"]))
            except _KEY_TYPE_VALUE_ERRORS:
                continue
        return out
