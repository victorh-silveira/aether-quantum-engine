"""Expoente de Hurst e variance ratio para persistencia de tendencia."""

from __future__ import annotations

import numpy as np


def hurst_exponent(prices: np.ndarray, window: int, *, min_window: int) -> np.ndarray:
    """Estima Hurst por janela deslizante; fallback 0.5 para series curtas."""
    n = len(prices)
    out = np.full(n, 0.5, dtype=np.float64)
    w = max(int(min_window), int(window))
    if n < w + 2:
        return out
    for i in range(w, n):
        segment = prices[i - w + 1 : i + 1].astype(np.float64)
        returns = np.diff(segment)
        if returns.size == 0 or np.std(returns) < 1e-12:
            continue
        mean_r = np.mean(returns)
        cumulative = np.cumsum(returns - mean_r)
        r = float(np.max(cumulative) - np.min(cumulative))
        s = float(np.std(returns, ddof=1))
        rs = r / max(s, 1e-12)
        out[i] = float(np.clip(np.log(rs + 1e-12) / np.log(float(w)), 0.0, 1.0))
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
