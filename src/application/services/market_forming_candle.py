"""Rastreamento em tempo real da vela M5 em formacao (ao-vivo)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from src.domain.models.market_data import Candle


@dataclass(frozen=True)
class LiveCandleSnapshot:
    """Snapshot completo de microestrutura da vela ao-vivo em formacao."""

    symbol: str
    epoch: int
    open_price: float
    high_price: float
    low_price: float
    spot_price: float
    elapsed_seconds: int
    remaining_seconds: int
    progress_pct: float
    current_side: str
    body_span: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    tick_velocity: float
    tick_acceleration: float


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Converte valor para float de forma segura e finita."""
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _extract_forming_candle_from_stream(stream: Any, symbol: str) -> Candle | None:
    """Extrai a ultima vela do buffer do stream (que representa a vela em formacao)."""
    if stream is None:
        return None
    store = getattr(stream, "micro_candles", None)
    if not isinstance(store, dict):
        return None
    history = store.get(str(symbol))
    if not isinstance(history, list) or len(history) == 0:
        return None
    last_item = history[-1]
    return last_item if isinstance(last_item, Candle) else None


def build_live_forming_candle_snapshot(
    orch: Any | None,
    symbol: str,
    *,
    boundary_seconds: int = 300,
    current_timestamp: float | None = None,
) -> LiveCandleSnapshot | None:
    """Constroi snapshot detalhado da vela ao-vivo a partir do stream e tick buffer."""
    if orch is None or not symbol:
        return None
    stream = getattr(orch, "stream", None)
    if stream is None:
        return None
    forming = _extract_forming_candle_from_stream(stream, symbol)
    if forming is None:
        return None
    buffer = getattr(stream, "tick_buffer", None)
    latest_tick = buffer.latest_price(str(symbol)) if buffer is not None else None
    open_px = float(forming.open)
    high_px = float(forming.high)
    low_px = float(forming.low)
    spot_px = (
        float(latest_tick) if latest_tick is not None and math.isfinite(float(latest_tick)) else float(forming.close)
    )
    high_px = max(high_px, spot_px)
    low_px = min(low_px, spot_px)
    now_ts = float(current_timestamp if current_timestamp is not None else time.time())
    candle_epoch = int(forming.epoch)
    elapsed = max(0, min(int(boundary_seconds), int(now_ts - candle_epoch)))
    remaining = max(0, int(boundary_seconds) - elapsed)
    progress = max(0.0, min(1.0, float(elapsed) / float(max(1, boundary_seconds))))
    delta = spot_px - open_px
    if delta > 1e-9:
        side = "CALL"
    elif delta < -1e-9:
        side = "PUT"
    else:
        side = "DOJI"
    span = high_px - low_px
    if span > 1e-12:
        upper_wick = (high_px - max(open_px, spot_px)) / span
        lower_wick = (min(open_px, spot_px) - low_px) / span
    else:
        upper_wick = 0.0
        lower_wick = 0.0
    vel, accel = 0.0, 0.0
    if buffer is not None and hasattr(buffer, "forming_bar_micro_stats"):
        stats = buffer.forming_bar_micro_stats(str(symbol))
        vel = _safe_float(getattr(stats, "price_velocity", 0.0))
        accel = _safe_float(getattr(stats, "price_acceleration", 0.0))
    return LiveCandleSnapshot(
        symbol=str(symbol).upper(),
        epoch=candle_epoch,
        open_price=open_px,
        high_price=high_px,
        low_price=low_px,
        spot_price=spot_px,
        elapsed_seconds=elapsed,
        remaining_seconds=remaining,
        progress_pct=progress,
        current_side=side,
        body_span=abs(delta),
        upper_wick_ratio=max(0.0, min(1.0, upper_wick)),
        lower_wick_ratio=max(0.0, min(1.0, lower_wick)),
        tick_velocity=vel,
        tick_acceleration=accel,
    )
