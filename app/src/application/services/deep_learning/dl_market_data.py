"""Leitura de series OHLC e microestrutura do stream Deriv."""

import time

import numpy as np


def load_symbol_close_ohlc(
    orch,
    symbol: str,
    *,
    timeframe: str = "micro",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Retorna close e open/high/low quando o buffer local tem o mesmo comprimento."""
    stream = orch.stream
    use_micro = str(timeframe).strip().lower() != "macro"
    getter = getattr(stream, "get_micro_numpy_series", None) if use_micro else None
    if use_micro and callable(getter):
        close = getter(symbol, "close")
        open_ = getter(symbol, "open")
        high = getter(symbol, "high")
        low = getter(symbol, "low")
    else:
        close = stream.get_numpy_series(symbol, "close")
        if len(close) == 0:
            return close, None, None, None
        open_ = stream.get_numpy_series(symbol, "open")
        high = stream.get_numpy_series(symbol, "high")
        low = stream.get_numpy_series(symbol, "low")
    if len(close) == 0:
        return close, None, None, None
    n = len(close)
    if len(open_) != n or len(high) != n or len(low) != n:
        return close, None, None, None
    return close, open_, high, low


def load_symbol_microstructure(orch, symbol: str, length: int) -> dict[str, np.ndarray] | None:
    """Retorna arrays de microestrutura alinhados ao historico de velas."""
    buffer = getattr(orch.stream, "tick_buffer", None)
    if buffer is None:
        return None
    return buffer.microstructure_arrays(symbol, length)


def slice_ohlc_window(
    close: np.ndarray,
    open_: np.ndarray | None,
    high: np.ndarray | None,
    low: np.ndarray | None,
    *,
    start: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Recorta janelas alinhadas a partir de start."""
    trimmed = close[start:]
    if open_ is None or high is None or low is None:
        return trimmed, None, None, None
    return trimmed, open_[start:], high[start:], low[start:]


def closed_model_ohlc(
    orch,
    symbol: str,
    close: np.ndarray,
    open_: np.ndarray | None,
    high: np.ndarray | None,
    low: np.ndarray | None,
    *,
    granularity: int,
    now_epoch: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Exclui a vela M5 em formacao para prever o contrato da vela corrente."""
    history = getattr(getattr(orch, "stream", None), "micro_candles", None)
    candles = history.get(symbol) if isinstance(history, dict) else None
    if not isinstance(candles, list) or len(candles) < 2:
        return close, open_, high, low
    if len(candles) != len(close):
        return close[:0], None, None, None
    now = int(time.time()) if now_epoch is None else int(now_epoch)
    last_epoch = int(candles[-1].epoch)
    if not last_epoch <= now < last_epoch + max(1, int(granularity)):
        return close, open_, high, low
    return (
        close[:-1],
        None if open_ is None else open_[:-1],
        None if high is None else high[:-1],
        None if low is None else low[:-1],
    )
