"""Modelos de dados para ticks e velas OHLC."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Tick:
    """Amostra de cotacao em um instante."""

    symbol: str
    quote: float
    time: datetime
    epoch: int


@dataclass
class Candle:
    """Vela OHLC em um periodo."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    time: datetime
    epoch: int


@dataclass
class MarketSeries:
    """Serie de ticks e velas por simbolo."""

    symbol: str
    ticks: list[Tick] = field(default_factory=list)
    candles: list[Candle] = field(default_factory=list)

    def records(self, data_type: str = "ticks") -> list[dict]:
        """Retorna lista de dicts para ticks ou candles."""
        if data_type == "ticks":
            return [{"time": t.time, "quote": t.quote} for t in self.ticks]
        return [
            {
                "time": c.time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
            }
            for c in self.candles
        ]
