"""Rotulos binarios alinhados a duracao do contrato Rise/Fall."""

from __future__ import annotations

import numpy as np


def binary_label_at_index(prices: np.ndarray, index: int, horizon_bars: int) -> bool:
    """Retorna True se close[index + horizon] > close[index]."""
    horizon = max(1, int(horizon_bars))
    j = index + horizon
    if j >= len(prices):
        return False
    return float(prices[j]) > float(prices[index])


def sequence_labels(prices: np.ndarray, lookback: int, horizon_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Gera targets binarios e mascara ativa para indices validos."""
    n = len(prices)
    horizon = max(1, int(horizon_bars))
    last_i = n - horizon - 1
    if n < lookback + horizon + 1 or last_i < lookback:
        return np.empty((0,)), np.empty((0,))
    targets = []
    masks = []
    for i in range(lookback, last_i + 1):
        targets.append(1.0 if binary_label_at_index(prices, i, horizon) else 0.0)
        masks.append(1.0)
    return np.array(targets, dtype=np.float32), np.array(masks, dtype=np.float32)
