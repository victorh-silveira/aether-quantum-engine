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

    def _resolve_fetch_count(self) -> int:
        """Define quantas velas buscar na sincronizacao inicial."""
        if "fetch_count" in self.config:
            return max(1, int(self.config["fetch_count"]))
        history_bars = int(self.config.get("history_bars", 0))
        if history_bars > 0:
            warmup = int(self.config.get("history_warmup_bars", 32))
            return history_bars + warmup
        return 500

    async def start_candle_stream(self, callback):
        """Ativa subscrições do cluster e busca histórico paralelo."""
        fetch_count = self._resolve_fetch_count()
        self.is_synchronized = False

        if not self.ws.is_running:
            raise ConnectionError("STREAM: WebSocket desconectado antes da sincronização.")

        self.logger.debug(f"DATA: Sincronizando Enxame Aegis ({len(self.symbols)} pares - OHLC {self.granularity}s)...")
        tasks = [self._fetch_symbol_history(s, fetch_count) for s in self.symbols]
        await asyncio.gather(*tasks)
        if self.symbols:
            bars = len(self.candles.get(self.symbols[0], []))
            self.logger.info(
                "DATA: Buffer pronto | %d simbolos | %d velas",
                len(self.symbols),
                bars,
            )
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
        """Trabalhador interno para busca paralela de historico com paginacao."""
        chunk_size = max(1, int(self.config.get("history_fetch_chunk", 500)))
        target = max(1, int(count))
        merged: list[Candle] = []
        end: str | int = "latest"
        while len(merged) < target:
            need = min(chunk_size, target - len(merged))
            request = {
                "ticks_history": symbol,
                "end": end,
                "style": "candles",
                "granularity": self.granularity,
                "count": need,
            }
            res = await self.ws.send(request)
            history = res.get("candles", [])
            if not history:
                break
            batch = [
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
            if merged:
                oldest_new = batch[0].epoch
                merged = [c for c in merged if c.epoch > oldest_new]
            merged = batch + merged
            if len(history) < need:
                break
            end = int(history[0]["epoch"]) - 1
        if len(merged) > target:
            merged = merged[-target:]
        self.candles[symbol] = merged
        self.logger.debug("DATA: Historico %s | %d velas (alvo %d)", symbol, len(merged), target)

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
