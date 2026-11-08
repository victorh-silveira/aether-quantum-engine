"""Normalizacao causal online para o vetor TCN ortogonal."""

from __future__ import annotations

import numpy as np


def center_unit_interval(series: np.ndarray) -> np.ndarray:
    """Mapeia [0, 1] para [-1, 1] com clip."""
    arr = np.asarray(series, dtype=np.float64)
    return np.clip((arr - 0.5) * 2.0, -1.0, 1.0)


def causal_robust_scale(
    series: np.ndarray,
    *,
    window: int = 288,
    clip: float = 3.0,
    min_hist: int = 8,
) -> np.ndarray:
    """Median/IQR rolling em [t-W, t-1] sem lookahead; clip em +/- clip."""
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    span = max(1, int(window))
    bound = float(clip)
    floor = max(2, int(min_hist))
    for i in range(n):
        start = max(0, i - span)
        hist = x[start:i]
        if len(hist) < floor:
            out[i] = 0.0
            continue
        med = float(np.median(hist))
        q75 = float(np.percentile(hist, 75))
        q25 = float(np.percentile(hist, 25))
        iqr = max(q75 - q25, 1e-8)
        out[i] = float(np.clip((x[i] - med) / iqr, -bound, bound))
    return out


def apply_causal_column_scale(
    matrix: np.ndarray,
    unbounded_cols: tuple[int, ...],
    *,
    window: int = 288,
    clip: float = 3.0,
) -> np.ndarray:
    """Aplica causal_robust_scale apenas nas colunas unbounded."""
    out = np.asarray(matrix, dtype=np.float32).copy()
    if out.ndim != 2 or out.size == 0:
        return out
    for col in unbounded_cols:
        if col < 0 or col >= out.shape[1]:
            continue
        scaled = causal_robust_scale(out[:, col], window=window, clip=clip)
        out[:, col] = scaled.astype(np.float32)
    return out
