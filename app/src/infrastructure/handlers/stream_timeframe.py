"""Granularidades macro/micro e subscricoes duplas de velas Deriv."""

from __future__ import annotations

import asyncio
from typing import Any

from src.infrastructure.api.deriv_granularity import normalize_granularity_seconds


def resolve_dual_granularity(data_config: dict) -> tuple[int, int]:
    """Resolve granularidade macro (DL) e micro (loop operacional)."""
    macro = normalize_granularity_seconds(int(data_config.get("granularity", 60)))
    micro_raw = int(data_config.get("micro_granularity", data_config.get("cycle_granularity", 60)))
    micro = normalize_granularity_seconds(micro_raw)
    return macro, micro


def resolve_mini_granularity(data_config: dict) -> int:
    """Resolve granularidade MINI (tape curto; default 60s)."""
    raw = int(data_config.get("mini_granularity", 60))
    return normalize_granularity_seconds(raw)


def resolve_triple_granularity(data_config: dict) -> tuple[int, int, int]:
    """Resolve MACRO/MICRO/MINI em segundos."""
    macro, micro = resolve_dual_granularity(data_config)
    return macro, micro, resolve_mini_granularity(data_config)


def ohlc_payload_granularity(ohlc: dict, macro: int, micro: int, mini: int | None = None) -> int:
    """Identifica granularidade de um payload OHLC da Deriv."""
    raw = ohlc.get("granularity")
    if raw is not None:
        return normalize_granularity_seconds(int(raw))
    epoch = int(ohlc["open_time"])
    if epoch % macro == 0:
        return macro
    if epoch % micro == 0:
        return micro
    if mini is not None:
        return normalize_granularity_seconds(int(mini))
    return micro


def resolve_micro_fetch_count(data_config: dict) -> int:
    """Quantidade de velas micro a sincronizar para o relogio operacional."""
    if "micro_fetch_count" in data_config:
        return max(1, int(data_config["micro_fetch_count"]))
    micro_bars = int(data_config.get("micro_history_bars", 0))
    if micro_bars > 0:
        return micro_bars
    startup = int(data_config.get("startup_fetch_bars", 512))
    return max(64, min(startup, 512))


def resolve_mini_fetch_count(data_config: dict) -> int:
    """Quantidade de velas MINI a sincronizar."""
    if "mini_fetch_count" in data_config:
        return max(1, int(data_config["mini_fetch_count"]))
    mini_bars = int(data_config.get("mini_history_bars", 0))
    if mini_bars > 0:
        return mini_bars
    startup = int(data_config.get("startup_fetch_bars", 512))
    return max(64, min(startup, 1024))


def granularity_label(seconds: int) -> str:
    """Rotulo curto de timeframe a partir da granularidade em segundos."""
    sec = max(1, int(seconds))
    if sec >= 86400:
        return f"D{sec // 86400}"
    if sec >= 3600:
        return f"H{sec // 3600}"
    if sec >= 60:
        return f"M{sec // 60}"
    return f"S{sec}"


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
