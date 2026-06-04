"""Leitura de series OHLC do stream Deriv para treino e inferencia."""

import numpy as np


def load_symbol_close_ohlc(
    orch, symbol: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Retorna close e open/high/low quando o buffer local tem o mesmo comprimento."""
    stream = orch.stream
    close = stream.get_numpy_series(symbol, "close")
    if len(close) == 0:
        return close, None, None, None
    open_ = stream.get_numpy_series(symbol, "open")
    high = stream.get_numpy_series(symbol, "high")
    low = stream.get_numpy_series(symbol, "low")
    n = len(close)
    if len(open_) != n or len(high) != n or len(low) != n:
        return close, None, None, None
    return close, open_, high, low


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
