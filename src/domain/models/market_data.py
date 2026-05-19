"""Modelos de dados para ticks e velas OHLC."""

from dataclasses import dataclass
from datetime import datetime


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
