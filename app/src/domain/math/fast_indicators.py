"""Indicadores matematicos causais de alta velocidade com aceleracao Numba JIT e fallback NumPy."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


try:
    numba = importlib.import_module("numba")
    njit = numba.njit
    _HAS_NUMBA = True
except (ImportError, AttributeError):
    numba = None
    njit = None
    _HAS_NUMBA = False


def is_numba_available() -> bool:
    """Indica se a biblioteca Numba JIT esta presente no ambiente."""
    return bool(_HAS_NUMBA)


def _ema_core_python(values: np.ndarray, alpha: float) -> np.ndarray:
    """Calcula iteracao causal de EMA."""
    n = values.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = values[0]
    for i in range(1, n):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _true_range_core_python(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Calcula iteracao causal de True Range."""
    n = high.shape[0]
    tr = np.empty(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_cp = abs(high[i] - close[i - 1])
        l_cp = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_cp, l_cp)
    return tr


_ema_core: Any = _ema_core_python
_true_range_core: Any = _true_range_core_python

if _HAS_NUMBA and njit is not None:
    _ema_core = njit(fastmath=True, nogil=True)(_ema_core_python)
    _true_range_core = njit(fastmath=True, nogil=True)(_true_range_core_python)


def fast_ema(values: np.ndarray | list[float], span: int) -> np.ndarray:
    """Calcula Media Movel Exponencial causal de forma vetorizada."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.empty(0, dtype=np.float64)
    valid_span = max(1, int(span))
    alpha = 2.0 / (valid_span + 1.0)
    return _ema_core(arr, float(alpha))


def fast_true_range(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
) -> np.ndarray:
    """Calcula True Range causal para series OHLC."""
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    n = min(high_arr.size, low_arr.size, close_arr.size)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    return _true_range_core(high_arr[:n], low_arr[:n], close_arr[:n])


def fast_atr(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    period: int = 14,
) -> np.ndarray:
    """Calcula Average True Range causal via suavizacao exponencial."""
    tr = fast_true_range(high, low, close)
    if tr.size == 0:
        return np.empty(0, dtype=np.float64)
    valid_period = max(1, int(period))
    return fast_ema(tr, span=valid_period)


def fast_log_returns(prices: np.ndarray | list[float]) -> np.ndarray:
    """Calcula retornos logaritmicos causais."""
    arr = np.asarray(prices, dtype=np.float64)
    if arr.size <= 1:
        return np.empty(0, dtype=np.float64)
    valid_mask = arr > 0.0
    if not np.all(valid_mask):
        arr = np.where(arr <= 0.0, 1e-12, arr)
    return np.diff(np.log(arr))
