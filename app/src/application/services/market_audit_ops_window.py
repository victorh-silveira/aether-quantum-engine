"""Janela ops: deslocamento liquido das ultimas N velas M1 fechadas."""

from __future__ import annotations

from math import isfinite
from typing import Any

from src.application.services.market_audit_candle import candle_binary_side
from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection


_EPS = 1e-12
_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


def closed_micro_candles(stream: Any, symbol: str) -> list[Candle]:
    """Velas micro ja fechadas (exclui a formando no fim do buffer)."""
    if stream is None:
        return []
    store = getattr(stream, "micro_candles", None)
    if not isinstance(store, dict):
        return []
    history = store.get(symbol)
    if not isinstance(history, list) or len(history) < 2:
        return []
    closed: list[Candle] = []
    for item in history[:-1]:
        if isinstance(item, Candle):
            closed.append(item)
    return closed


def ops_window_from_candles(candles: list[Candle], *, bars: int) -> tuple[str | None, float | None]:
    """Lado e corpo liquido open[0]→close[-1]; None se N incompleto."""
    n = max(1, int(bars))
    if len(candles) < n:
        return None, None
    window = candles[-n:]
    first = window[0]
    last = window[-1]
    try:
        open_px = float(first.open)
        close_px = float(last.close)
    except (TypeError, ValueError):
        return None, None
    if not isfinite(open_px) or not isfinite(close_px):
        return None, None
    synthetic = Candle(
        symbol=str(last.symbol),
        open=open_px,
        high=max(open_px, close_px),
        low=min(open_px, close_px),
        close=close_px,
        time=last.time,
        epoch=int(last.epoch),
    )
    side = candle_binary_side(synthetic)
    body = abs(close_px - open_px)
    dir_name = side if side in _VALID else None
    body_val = body if body > _EPS else None
    return dir_name, body_val


def ops_window_from_stream(stream: Any, symbol: str, *, bars: int) -> tuple[str | None, float | None, bool]:
    """Dir/body da janela ops; stamped False se buffer curto."""
    n = max(1, int(bars))
    closed = closed_micro_candles(stream, symbol)
    if len(closed) < n:
        return None, None, False
    side, body = ops_window_from_candles(closed, bars=n)
    return side, body, True


def ops_window_candle_side(metrics: dict[str, Any] | None) -> str | None:
    """Lado da janela ops (CALL/PUT); nao cai na M1 isolada."""
    if not isinstance(metrics, dict):
        return None
    side = str(metrics.get("ops_window_candle_dir") or "").strip().upper()
    return side if side in _VALID else None


def ops_window_candle_body(metrics: dict[str, Any] | None) -> float | None:
    """Corpo liquido da janela ops; None se ausente."""
    if not isinstance(metrics, dict):
        return None
    raw = metrics.get("ops_window_candle_body")
    if raw is None:
        return None
    try:
        body = float(raw)
    except (TypeError, ValueError):
        return None
    if not isfinite(body) or body < 0.0:
        return None
    return body


def ops_window_stamped(metrics: dict[str, Any] | None) -> bool:
    """True se SCALE stampou a janela N completa neste ciclo."""
    if not isinstance(metrics, dict):
        return False
    return bool(metrics.get("ops_window_stamped"))


def stamp_ops_window_metrics(
    metrics: dict[str, Any],
    stream: Any,
    symbol: str,
    *,
    bars: int,
) -> None:
    """Grava ops_window_* no metrics; fail-closed se N incompleto."""
    n = max(1, int(bars))
    metrics["ops_window_bars"] = n
    side, body, stamped = ops_window_from_stream(stream, str(symbol), bars=n)
    metrics["ops_window_stamped"] = stamped
    metrics["ops_window_candle_dir"] = side
    metrics["ops_window_candle_body"] = body
