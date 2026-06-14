"""Rotulos binarios alinhados a duracao do contrato Rise/Fall."""

from __future__ import annotations

import numpy as np


LABEL_MODE_SPOT = "spot_forward"
LABEL_MODE_MA_TREND = "ma_trend"


def _rolling_mean(prices: np.ndarray, index: int, window: int) -> float:
    """Media movel dos closes terminando na barra index."""
    span = max(1, int(window))
    start = max(0, index - span + 1)
    return float(np.mean(prices[start : index + 1]))


def _forward_mean(prices: np.ndarray, index: int, horizon_bars: int, smooth_bars: int) -> float | None:
    """Media dos closes forward apos horizon ou None se indice invalido."""
    smooth = max(1, int(smooth_bars))
    forward_start = index + max(1, int(horizon_bars))
    forward_end = forward_start + smooth
    if forward_end > len(prices):
        return None
    return float(np.mean(prices[forward_start:forward_end]))


def binary_label_at_index(
    prices: np.ndarray,
    index: int,
    horizon_bars: int,
    *,
    smooth_bars: int = 1,
    label_mode: str = LABEL_MODE_SPOT,
    ma_window: int = 5,
) -> bool:
    """Retorna True para CALL conforme modo spot_forward ou ma_trend."""
    forward = _forward_mean(prices, index, horizon_bars, smooth_bars)
    if forward is None:
        return False
    mode = str(label_mode).strip().lower()
    if mode == LABEL_MODE_MA_TREND:
        current = _rolling_mean(prices, index, ma_window)
        return forward > current
    return forward > float(prices[index])


def sequence_labels(
    prices: np.ndarray,
    lookback: int,
    horizon_bars: int,
    *,
    smooth_bars: int = 1,
    label_mode: str = LABEL_MODE_SPOT,
    ma_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera targets binarios e mascara ativa para indices validos."""
    n = len(prices)
    horizon = max(1, int(horizon_bars))
    smooth = max(1, int(smooth_bars))
    tail = horizon + smooth
    last_i = n - horizon - smooth
    if n < lookback + tail or last_i < lookback:
        return np.empty((0,)), np.empty((0,))
    targets = []
    masks = []
    for i in range(lookback, last_i + 1):
        up = binary_label_at_index(
            prices,
            i,
            horizon,
            smooth_bars=smooth,
            label_mode=label_mode,
            ma_window=ma_window,
        )
        targets.append(1.0 if up else 0.0)
        masks.append(1.0)
    return np.array(targets, dtype=np.float32), np.array(masks, dtype=np.float32)
