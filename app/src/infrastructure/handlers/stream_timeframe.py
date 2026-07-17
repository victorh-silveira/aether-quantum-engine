"""Granularidades macro/micro e subscricoes duplas de velas Deriv."""

from __future__ import annotations

import asyncio
from typing import Any

from src.infrastructure.api.deriv_granularity import normalize_granularity_seconds


def resolve_dual_granularity(data_config: dict) -> tuple[int, int]:
    """Resolve granularidade macro (DL M15) e micro (loop operacional M5)."""
    macro = normalize_granularity_seconds(int(data_config.get("granularity", 900)))
    micro_raw = int(data_config.get("micro_granularity", data_config.get("cycle_granularity", 300)))
    micro = normalize_granularity_seconds(micro_raw)
    return macro, micro


def ohlc_payload_granularity(ohlc: dict, macro: int, micro: int) -> int:
    """Identifica granularidade de um payload OHLC da Deriv."""
    raw = ohlc.get("granularity")
    if raw is not None:
        return normalize_granularity_seconds(int(raw))
    epoch = int(ohlc["open_time"])
    if epoch % macro == 0:
        return macro
    return micro


def resolve_micro_fetch_count(data_config: dict) -> int:
    """Quantidade de velas M5 a sincronizar para o relogio operacional."""
    if "micro_fetch_count" in data_config:
        return max(1, int(data_config["micro_fetch_count"]))
    micro_bars = int(data_config.get("micro_history_bars", 0))
    if micro_bars > 0:
        return micro_bars
    startup = int(data_config.get("startup_fetch_bars", 512))
    return max(64, min(startup, 512))


async def subscribe_candle_streams(ws: Any, symbols: list[str], granularity: int) -> None:
    """Assina fluxo OHLC para uma granularidade."""
    tasks = [
        ws.send(
            {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": int(granularity),
                "subscribe": 1,
                "end": "latest",
                "count": 1,
            }
        )
        for symbol in symbols
    ]
    await asyncio.gather(*tasks)


async def subscribe_tick_streams(ws: Any, symbols: list[str]) -> None:
    """Assina fluxo de ticks para microestrutura macro."""
    tasks = [
        ws.send(
            {
                "ticks_history": symbol,
                "style": "ticks",
                "subscribe": 1,
                "end": "latest",
                "count": 1,
            }
        )
        for symbol in symbols
    ]
    await asyncio.gather(*tasks)
