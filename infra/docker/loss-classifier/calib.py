from __future__ import annotations

from typing import Any

import numpy as np

from runtime import predict_p_loss


_TEMP_GRID = (0.70, 0.85, 1.00, 1.15, 1.30, 1.50)
_CAL_MIN_N = 32


def binary_nll(probs: list[float], labels: list[int]) -> float:
    arr = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    y = np.asarray(labels, dtype=np.float64)
    if arr.size == 0 or arr.size != y.size:
        return float("inf")
    return float(-np.mean(y * np.log(arr) + (1.0 - y) * np.log(1.0 - arr)))


def binary_ece(probs: list[float], labels: list[int], *, n_bins: int = 10) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if arr.size < 2 or arr.size != y.size:
        return 1.0
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n = float(arr.size)
    for i in range(int(n_bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i + 1 == int(n_bins):
            mask = (arr >= lo) & (arr <= hi)
        else:
            mask = (arr >= lo) & (arr < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(arr[mask]))
        ece += (float(count) / n) * abs(acc - conf)
    return float(ece)


def fit_temperature(model: Any, buffer_x: list[list[float]], buffer_y: list[int]) -> float:
    if len(buffer_y) < int(_CAL_MIN_N) or len(buffer_x) != len(buffer_y):
        return 1.0
    best_t = 1.0
    best_nll = float("inf")
    for temp in _TEMP_GRID:
        probs = [predict_p_loss(model, row, temperature=float(temp)) for row in buffer_x]
        nll = binary_nll(probs, buffer_y)
        if nll < best_nll:
            best_nll = nll
            best_t = float(temp)
    return float(best_t)


def buffer_p_loss(model: Any, buffer_x: list[list[float]], *, temperature: float) -> list[float]:
    return [predict_p_loss(model, row, temperature=float(temperature)) for row in buffer_x]
