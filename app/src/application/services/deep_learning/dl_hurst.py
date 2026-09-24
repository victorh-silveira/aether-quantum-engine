"""Expoente de Hurst e variance ratio para persistencia de tendencia."""

from __future__ import annotations

import numpy as np


def hurst_exponent(prices: np.ndarray, window: int, *, min_window: int) -> np.ndarray:
    """Estima H local por escala MSD causal; 0.5 quando ha dados insuficientes."""
    n = len(prices)
    out = np.full(n, 0.5, dtype=np.float64)
    w = max(int(min_window), int(window))
    if n < w + 2:
        return out
    lags = np.asarray([lag for lag in (1, 2, 4, 8) if lag <= w // 4], dtype=np.int64)
    if len(lags) < 2:
        return out
    log_lags = np.log(lags.astype(np.float64))
    for i in range(w - 1, n):
        segment = np.log(np.maximum(np.asarray(prices[i - w + 1 : i + 1], dtype=np.float64), 1e-12))
        msd = np.asarray([np.mean((segment[lag:] - segment[:-lag]) ** 2) for lag in lags])
        if not np.all(np.isfinite(msd)) or np.any(msd <= 1e-16):
            continue
        slope = float(np.polyfit(log_lags, np.log(msd), deg=1)[0])
        out[i] = float(np.clip(0.5 * slope, 0.0, 1.0))
    return out


def variance_ratio(prices: np.ndarray, short: int, long: int) -> np.ndarray:
    """Proxy de persistencia: variancia de retornos longos vs curtos."""
    n = len(prices)
    out = np.ones(n, dtype=np.float64)
    s = max(2, int(short))
    lg = max(s + 1, int(long))
    if n < lg + 2:
        return out
    returns = np.diff(prices.astype(np.float64))
    for i in range(lg, len(returns)):
        short_slice = returns[i - s + 1 : i + 1]
        long_slice = returns[i - lg + 1 : i + 1]
        var_s = float(np.var(short_slice))
        var_l = float(np.var(long_slice))
        if var_s < 1e-14:
            continue
        out[i + 1] = float(np.clip(var_l / (var_s * (lg / s)), 0.0, 3.0))
    return out
