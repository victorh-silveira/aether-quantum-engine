"""Busca velas OHLC pela API sem alterar buffer local."""

from __future__ import annotations

import logging
from typing import Any


_KEY_TYPE_VALUE_ERRORS = (KeyError, TypeError, ValueError)


async def _send_candle_history(
    ws: Any,
    symbol: str,
    granularity: int,
    count: int,
    logger: logging.Logger,
    log_tag: str,
) -> list[dict]:
    """Envia ticks_history e retorna lista bruta de velas."""
    if count <= 0 or not ws.is_running:
        return []
    req = {
        "ticks_history": symbol,
        "end": "latest",
        "style": "candles",
        "granularity": granularity,
        "count": count,
    }
    try:
        res = await ws.send(req)
    except Exception as e:
        logger.debug("DATA: OHLC %s %s g=%s: %s", log_tag, symbol, granularity, e)
        return []
    if res.get("error"):
        return []
    return res.get("candles") or []


async def fetch_candle_ohlc_rows(
    ws: Any,
    symbol: str,
    granularity: int,
    count: int,
    logger: logging.Logger,
) -> list[tuple[float, float, float, float]]:
    """Retorna tuplas OHLC de velas remotas."""
    history = await _send_candle_history(ws, symbol, granularity, count, logger, "completo")
    out: list[tuple[float, float, float, float]] = []
    for c in history:
        try:
            out.append((float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])))
        except _KEY_TYPE_VALUE_ERRORS:
            continue
    return out


async def fetch_candle_close_rows(
    ws: Any,
    symbol: str,
    granularity: int,
    count: int,
    logger: logging.Logger,
) -> list[float]:
    """Retorna fechamentos de velas remotas."""
    history = await _send_candle_history(ws, symbol, granularity, count, logger, "historia extra")
    out: list[float] = []
    for c in history:
        try:
            out.append(float(c["close"]))
        except _KEY_TYPE_VALUE_ERRORS:
            continue
    return out
