"""Paginacao resiliente de ticks_history com throttling e retry."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.domain.models.market_data import Candle


if TYPE_CHECKING:
    from src.infrastructure.api.websocket_manager import WebSocketManager


def parse_history_fetch_config(config: dict) -> dict[str, float | int]:
    """Le parametros de throttling e retry para ticks_history."""
    return {
        "chunk_size": max(1, int(config.get("history_fetch_chunk", 500))),
        "chunk_delay": max(0.0, float(config.get("history_fetch_delay_seconds", 0.35))),
        "symbol_delay": max(0.0, float(config.get("history_fetch_symbol_delay_seconds", 2.0))),
        "max_retries": max(1, int(config.get("history_fetch_rate_limit_retries", 12))),
        "backoff_base": max(1.0, float(config.get("history_fetch_rate_limit_backoff", 2.0))),
        "backoff_cap": max(1.0, float(config.get("history_fetch_rate_limit_max_delay", 45.0))),
    }


def is_rate_limit_error(payload: dict) -> bool:
    """Indica erro de rate limit da Deriv em resposta ticks_history."""
    err = payload.get("error")
    if not err:
        return False
    if isinstance(err, dict):
        msg = str(err.get("message", ""))
        code = str(err.get("code", ""))
    else:
        msg = str(err)
        code = ""
    text = f"{msg} {code}".lower()
    return "rate limit" in text


def candles_from_payload(symbol: str, history: list[dict]) -> list[Candle]:
    """Converte candles da API em objetos Candle."""
    batch: list[Candle] = []
    for row in history:
        batch.append(
            Candle(
                symbol=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                time=datetime.fromtimestamp(row["epoch"]),
                epoch=row["epoch"],
            )
        )
    return batch


def merge_candle_pages(existing: list[Candle], batch: list[Candle]) -> list[Candle]:
    """Anexa pagina mais antiga evitando duplicatas de epoch."""
    if not batch:
        return existing
    if not existing:
        return batch
    oldest_new = batch[0].epoch
    trimmed = [c for c in existing if c.epoch > oldest_new]
    return batch + trimmed


async def fetch_paginated_candle_history(
    ws: WebSocketManager,
    *,
    symbol: str,
    granularity: int,
    target: int,
    fetch_cfg: dict[str, float | int],
    logger: logging.Logger,
    existing: list[Candle] | None = None,
) -> list[Candle]:
    """Busca historico OHLC paginado com delay entre paginas e retry em rate limit."""
    merged = list(existing or [])
    goal = max(1, int(target))
    if len(merged) >= goal:
        return merged[-goal:]
    chunk_size = int(fetch_cfg["chunk_size"])
    end: str | int = int(merged[0].epoch) - 1 if merged else "latest"
    chunk_delay = float(fetch_cfg["chunk_delay"])
    max_retries = int(fetch_cfg["max_retries"])
    backoff_base = float(fetch_cfg["backoff_base"])
    backoff_cap = float(fetch_cfg["backoff_cap"])

    while len(merged) < goal:
        need = min(chunk_size, goal - len(merged))
        request = {
            "ticks_history": symbol,
            "end": end,
            "style": "candles",
            "granularity": granularity,
            "count": need,
        }
        res: dict[str, Any] | None = None
        for attempt in range(max_retries + 1):
            res = await ws.send(request)
            if not res.get("error"):
                break
            if is_rate_limit_error(res) and attempt < max_retries:
                delay = min(backoff_cap, chunk_delay * (backoff_base ** (attempt + 1)))
                logger.warning(
                    "DATA: rate limit %s | retry %d/%d em %.1fs",
                    symbol,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            logger.error(
                "DATA: Historico %s falhou | granularity=%ss | %s",
                symbol,
                granularity,
                res["error"].get("message", res["error"]) if isinstance(res["error"], dict) else res["error"],
            )
            return merged[-goal:] if len(merged) > goal else merged

        history = res.get("candles", []) if res else []
        if not history:
            break
        batch = candles_from_payload(symbol, history)
        merged = merge_candle_pages(merged, batch)
        if len(history) < need:
            break
        end = int(history[0]["epoch"]) - 1
        if chunk_delay > 0:
            await asyncio.sleep(chunk_delay)

    if len(merged) > goal:
        merged = merged[-goal:]
    return merged
