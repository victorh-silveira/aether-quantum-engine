"""Features causais de velas fechadas por ativo e periodo."""

from __future__ import annotations

import math

from indicator.domain import Candle

LOOKBACK = 8


def features(candles: list[Candle]) -> tuple[float, ...]:
    """Retornos causais das ultimas oito velas."""
    if len(candles) < LOOKBACK + 1:
        raise ValueError("historico insuficiente")
    closes = [candle.close for candle in candles[-LOOKBACK - 1 :]]
    if any(close <= 0 or not math.isfinite(close) for close in closes):
        raise ValueError("preco invalido")
    return tuple(math.log(closes[index + 1] / closes[index]) for index in range(LOOKBACK))


def ohlc_features(candles: list[Candle]) -> tuple[float, ...]:
    """Retornos, corpo e amplitude das ultimas oito velas fechadas."""
    returns = features(candles)
    recent = candles[-LOOKBACK:]
    if any(candle.open <= 0 or candle.high < candle.low for candle in recent):
        raise ValueError("OHLC invalido")
    bodies = tuple((candle.close - candle.open) / candle.open for candle in recent)
    ranges = tuple((candle.high - candle.low) / candle.open for candle in recent)
    return returns + bodies + ranges
