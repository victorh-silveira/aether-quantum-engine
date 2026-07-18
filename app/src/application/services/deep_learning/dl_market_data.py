"""Leitura de series OHLC e microestrutura do stream Deriv."""

import numpy as np


def load_symbol_close_ohlc(
    orch,
    symbol: str,
    *,
    timeframe: str = "macro",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Retorna close e open/high/low quando o buffer local tem o mesmo comprimento."""
    stream = orch.stream
    use_micro = str(timeframe).strip().lower() == "micro"
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
