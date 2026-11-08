"""Injeta tick live na vela OHLC e na ultima linha de microestrutura."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


logger = logging.getLogger("AETH")

_MICRO_FIELDS = (
    "tick_count",
    "mean_inter_tick_ms",
    "price_velocity",
    "price_acceleration",
    "consecutive_diff_std",
    "micro_bid_ask_spread_momentum",
    "volatility_shadow_ratio",
)


def store_patched_ohlc_snapshot(
    orch: Any,
    symbol: str,
    close: np.ndarray,
    open_: np.ndarray | None,
    high: np.ndarray | None,
    low: np.ndarray | None,
) -> None:
    """Guarda OHLC patchado no orquestrador para SCALE/flow no mesmo ciclo."""
    bag = getattr(orch, "_patched_ohlc", None)
    if not isinstance(bag, dict):
        bag = {}
        orch._patched_ohlc = bag
    bag[str(symbol)] = {
        "close": np.asarray(close, dtype=np.float64),
        "open": None if open_ is None else np.asarray(open_, dtype=np.float64),
        "high": None if high is None else np.asarray(high, dtype=np.float64),
        "low": None if low is None else np.asarray(low, dtype=np.float64),
    }


def get_patched_ohlc_snapshot(orch: Any, symbol: str) -> dict[str, np.ndarray | None] | None:
    """Retorna snapshot OHLC patchado do simbolo, se existir."""
    bag = getattr(orch, "_patched_ohlc", None)
    if not isinstance(bag, dict):
        return None
    snap = bag.get(str(symbol))
    return snap if isinstance(snap, dict) else None


def patch_forming_bar_with_live_tick(
    orch: Any,
    symbol: str,
    close: np.ndarray,
    open_: np.ndarray | None,
    high: np.ndarray | None,
    low: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Atualiza close/high/low da ultima barra com o ultimo tick; fail-open sem tick."""
    if close is None or len(close) == 0:
        return close, open_, high, low
    stream = getattr(orch, "stream", None)
    buffer = getattr(stream, "tick_buffer", None) if stream is not None else None
    if buffer is None or not hasattr(buffer, "latest_price"):
        return close, open_, high, low
    tick = buffer.latest_price(str(symbol))
    if tick is None:
        return close, open_, high, low
    price = float(tick)
    if not np.isfinite(price):
        return close, open_, high, low
    close_was = float(close[-1])
    close_out = np.array(close, dtype=np.float64, copy=True)
    close_out[-1] = price
    high_out = None if high is None else np.array(high, dtype=np.float64, copy=True)
    low_out = None if low is None else np.array(low, dtype=np.float64, copy=True)
    open_out = None if open_ is None else np.array(open_, dtype=np.float64, copy=True)
    if high_out is not None and len(high_out) == len(close_out):
        high_out[-1] = max(float(high_out[-1]), price)
    if low_out is not None and len(low_out) == len(close_out):
        low_out[-1] = min(float(low_out[-1]), price)
    live_n = int(buffer.live_tick_count(str(symbol))) if hasattr(buffer, "live_tick_count") else 0
    logger.debug(
        "DL: tick_patch symbol=%s tick=%.5f close_was=%.5f live_n=%d",
        symbol,
        price,
        close_was,
        live_n,
    )
    return close_out, open_out, high_out, low_out


def patch_forming_bar_microstructure(
    orch: Any,
    symbol: str,
    micro: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray] | None:
    """Sobrescreve a ultima linha de microestrutura com stats dos ticks live."""
    if not isinstance(micro, dict) or not micro:
        return micro
    stream = getattr(orch, "stream", None)
    buffer = getattr(stream, "tick_buffer", None) if stream is not None else None
    if buffer is None or not hasattr(buffer, "forming_bar_micro_stats"):
        return micro
    if hasattr(buffer, "live_tick_count") and int(buffer.live_tick_count(str(symbol))) < 2:
        return micro
    stats = buffer.forming_bar_micro_stats(str(symbol))
    out: dict[str, np.ndarray] = {}
    values = {
        "tick_count": float(stats.tick_count),
        "mean_inter_tick_ms": float(stats.mean_inter_tick_ms),
        "price_velocity": float(stats.price_velocity),
        "price_acceleration": float(stats.price_acceleration),
        "consecutive_diff_std": float(stats.consecutive_diff_std),
        "micro_bid_ask_spread_momentum": float(stats.micro_bid_ask_spread_momentum),
        "volatility_shadow_ratio": float(stats.volatility_shadow_ratio),
    }
    for key in _MICRO_FIELDS:
        arr = micro.get(key)
        if arr is None or len(arr) == 0:
            continue
        copied = np.array(arr, dtype=np.float64, copy=True)
        copied[-1] = float(values[key])
        out[key] = copied
    for key, arr in micro.items():
        if key not in out:
            out[key] = arr
    return out if out else micro
