"""Telemetria do resultado da ultima vela micro fechada."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.application.services.log_dedupe import log_info_if_changed
from src.domain.models.market_data import Candle


_EPS = 1e-12


def candle_binary_side(candle: Candle) -> str:
    """CALL se close>open, PUT se close<open, DOJI se iguais."""
    delta = float(candle.close) - float(candle.open)
    if delta > _EPS:
        return "CALL"
    if delta < -_EPS:
        return "PUT"
    return "DOJI"


def last_closed_micro_candle(stream: Any, symbol: str) -> Candle | None:
    """Retorna a ultima vela micro ja fechada (penultima do buffer)."""
    if stream is None:
        return None
    store = getattr(stream, "micro_candles", None)
    if not isinstance(store, dict):
        return None
    history = store.get(symbol)
    if not isinstance(history, list) or len(history) < 2:
        return None
    candle = history[-2]
    return candle if isinstance(candle, Candle) else None


def closed_micro_candle_dir_from_stream(stream: Any, symbol: str) -> str | None:
    """Lado CALL/PUT da vela do [CANDLE]; None se ausente ou DOJI."""
    candle = last_closed_micro_candle(stream, symbol)
    if candle is None:
        return None
    side = candle_binary_side(candle)
    return side if side in {"CALL", "PUT"} else None


def closed_micro_candle_body_from_stream(stream: Any, symbol: str) -> float | None:
    """Corpo |c-o| da vela do [CANDLE]; None se ausente ou DOJI."""
    candle = last_closed_micro_candle(stream, symbol)
    if candle is None:
        return None
    body = abs(float(candle.close) - float(candle.open))
    if body <= _EPS:
        return None
    return body


def resolve_micro_granularity_seconds(orch: Any) -> int:
    """Resolve granularidade micro do stream ou settings (padrao 60)."""
    stream = getattr(orch, "stream", None) if orch is not None else None
    if stream is not None:
        raw = getattr(stream, "micro_granularity", None)
        if raw is not None:
            try:
                return max(1, int(raw))
            except (TypeError, ValueError):
                pass
    config = getattr(orch, "config", None) if orch is not None else None
    if isinstance(config, dict):
        data = config.get("data_handler")
        if isinstance(data, dict) and data.get("micro_granularity") is not None:
            try:
                return max(1, int(data["micro_granularity"]))
            except (TypeError, ValueError):
                pass
    return 60


def format_candle_outcome_line(
    symbol: str,
    candle: Candle,
    *,
    granularity: int = 300,
    start_epoch: int | None = None,
) -> str:
    """Linha [CANDLE] com lado OHLC e janela temporal da vela fechada."""
    side = candle_binary_side(candle)
    gran = max(1, int(granularity))
    s_ep = int(start_epoch) if start_epoch is not None else int(candle.epoch)
    e_ep = int(candle.epoch) + (gran if start_epoch is None else 60)
    start = datetime.fromtimestamp(s_ep, tz=UTC).astimezone()
    end = datetime.fromtimestamp(e_ep, tz=UTC).astimezone()
    tf = f"M{max(1, gran // 60)}" if gran % 60 == 0 else f"{gran}s"
    return (
        f"[CANDLE] || {tf} || {str(symbol).upper()}: {side} | "
        f"o={float(candle.open):.5f} c={float(candle.close):.5f} | "
        f"{start.strftime('%H:%M:%S')}->{end.strftime('%H:%M:%S')} | epoch={s_ep}"
    )


def _all_closed_micro_candles(stream: Any, symbol: str) -> list[Candle]:
    """Retorna lista de velas micro ja fechadas."""
    if stream is None:
        return []
    store = getattr(stream, "micro_candles", None)
    if not isinstance(store, dict):
        return []
    history = store.get(symbol)
    if not isinstance(history, list) or len(history) < 2:
        return []
    return [item for item in history[:-1] if isinstance(item, Candle)]


def log_closed_candle_outcomes(logger: Any, orch: Any, decisions: dict[str, Any]) -> None:
    """Emite [CANDLE] por simbolo do ciclo usando a janela de operacoes fechada."""
    if orch is None:
        return
    stream = getattr(orch, "stream", None)
    symbols = list(decisions.keys()) if isinstance(decisions, dict) else []
    if not symbols:
        symbols = list(getattr(orch, "symbols", []) or [])
    gran = resolve_micro_granularity_seconds(orch)
    for symbol in symbols:
        closed = _all_closed_micro_candles(stream, str(symbol))
        if not closed:
            continue
        candle = closed[-1]
        line = format_candle_outcome_line(str(symbol), candle, granularity=gran)
        log_key = str(int(candle.epoch))
        log_info_if_changed(
            orch,
            logger,
            f"candle_closed:{symbol}",
            log_key,
            "%s",
            line,
        )
