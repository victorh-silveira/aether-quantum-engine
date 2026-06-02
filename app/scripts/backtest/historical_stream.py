"""Stream sintetico de velas historicas para backtest com Gemini."""

from __future__ import annotations

from typing import Any

import numpy as np
from src.application.services.llm.context_runtime_support import tail_closes


M15_SECONDS = 900
M5_SECONDS = 300


class HistoricalStream:
    """Expoe fechamentos ate bar_index sem chamadas Deriv adicionais."""

    def __init__(self, m15: dict[str, list[float]], m5: dict[str, list[float]], *, bar_index: int = 0):
        self._m15 = m15
        self._m5 = m5
        self._bar_index = bar_index

    def set_bar_index(self, bar_index: int) -> None:
        self._bar_index = max(0, int(bar_index))

    def _series_upto(self, sym: str, granularity: int) -> list[float]:
        if granularity <= M5_SECONDS:
            series = self._m5.get(sym, [])
            end = min(len(series) - 1, max(0, (self._bar_index + 1) * 3 - 1))
        else:
            series = self._m15.get(sym, [])
            end = min(len(series) - 1, self._bar_index)
        if end < 0 or not series:
            return []
        chunk = [float(x) for x in series[: end + 1]]
        ratio = max(1, granularity // M15_SECONDS)
        if ratio > 1 and granularity > M5_SECONDS:
            sampled: list[float] = []
            for i in range(0, len(chunk), ratio):
                sampled.append(chunk[min(i + ratio - 1, len(chunk) - 1)])
            chunk = sampled or chunk
        return chunk

    async def fetch_candle_closes(self, sym: str, granularity: int, count: int) -> list[float]:
        chunk = self._series_upto(sym, granularity)
        return tail_closes(chunk, count)

    async def fetch_candle_ohlc(
        self, sym: str, granularity: int, count: int
    ) -> list[tuple[float, float, float, float]]:
        closes = await self.fetch_candle_closes(sym, granularity, count)
        rows: list[tuple[float, float, float, float]] = []
        for price in closes[-10:]:
            p = float(price)
            rows.append((p, p, p, p))
        return rows

    def get_numpy_series(self, symbol: str, field: str = "close") -> Any:
        series = self._m15.get(symbol, [])
        end = min(len(series) - 1, self._bar_index)
        if end < 0:
            return np.array([], dtype=np.float64)
        if field != "close":
            return np.array([], dtype=np.float64)
        return np.asarray(series[: end + 1], dtype=np.float64)
