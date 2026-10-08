"""Diferenciacao fracionaria causal para series temporais financeiras."""

from __future__ import annotations

import numpy as np


__all__ = (
    "get_fractional_weights",
    "frac_diff_causal",
)


def get_fractional_weights(
    d: float,
    size: int,
    *,
    threshold: float = 1e-4,
) -> np.ndarray:
    """Gera pesos fracionarios via expansao binomial com truncamento causal."""
    if size <= 0:
        return np.empty(0, dtype=np.float64)
    if abs(float(d)) < 1e-12:
        weights = np.zeros(size, dtype=np.float64)
        weights[0] = 1.0
        return weights

    weights = [1.0]
    for k in range(1, size):
        w_k = -weights[-1] * (float(d) - float(k) + 1.0) / float(k)
        if abs(w_k) < float(threshold):
            break
        weights.append(float(w_k))

    return np.asarray(weights, dtype=np.float64)


def frac_diff_causal(
    series: np.ndarray,
    d: float = 0.45,
    *,
    threshold: float = 1e-4,
) -> np.ndarray:
    """Calcula diferenciacao fracionaria estritamente causal preservando dimensao."""
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        if arr.ndim == 2:
            return np.column_stack(
                [frac_diff_causal(arr[:, col], d=d, threshold=threshold) for col in range(arr.shape[1])]
            )
        return np.copy(arr)

    if abs(float(d)) < 1e-12:
        return np.copy(arr)

    clean_arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    weights = get_fractional_weights(d, len(clean_arr), threshold=threshold)
    convolved = np.convolve(clean_arr, weights, mode="full")
    result = convolved[: len(clean_arr)]

    mask_nan = np.isnan(arr)
    if np.any(mask_nan):
        result[mask_nan] = np.nan

    return result
