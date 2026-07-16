"""Extracao de sequencias TCN e rotulos binarios a partir de precos OHLC."""

import numpy as np

from src.application.services.deep_learning.dl_feature_build import (
    FEATURE_DIM,
    precompute_price_series,
)
from src.application.services.deep_learning.dl_feature_matrix import build_feature_matrix
from src.application.services.deep_learning.dl_labels import sequence_labels


def extract_sequences(
    prices: np.ndarray,
    lookback: int,
    *,
    granularity: int = 60,
    label_horizon_bars: int = 1,
    label_smooth_bars: int = 1,
    label_mode: str = "ma_trend",
    label_ma_window: int = 5,
    implied_vol_bars: int = 60,
    symbol: str = "RDBULL",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrai tensores (N, L, F), rotulos binarios e mascara ativa."""
    n = len(prices)
    horizon = max(1, int(label_horizon_bars))
    smooth = max(1, int(label_smooth_bars))
    tail = horizon + smooth
    if n < lookback + tail:
        return np.empty((0, lookback, FEATURE_DIM)), np.empty((0,)), np.empty((0,))
    series = precompute_price_series(
        prices,
        granularity=granularity,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
        implied_vol_bars=implied_vol_bars,
    )
    feature_matrix = build_feature_matrix(series)
    targets, masks = sequence_labels(
        prices,
        lookback,
        horizon,
        smooth_bars=smooth,
        label_mode=label_mode,
        ma_window=label_ma_window,
    )
    count = len(targets)
    if count == 0:
        return np.empty((0, lookback, FEATURE_DIM)), np.empty((0,)), np.empty((0,))
    sequences = []
    for offset, i in enumerate(range(lookback, lookback + count)):
        sequences.append(feature_matrix[i - lookback + 1 : i + 1])
        _ = offset
    return (
        np.array(sequences, dtype=np.float32),
        targets,
        masks,
    )


def sequence_price_deltas(
    prices: np.ndarray,
    lookback: int,
    *,
    label_horizon_bars: int = 1,
    label_smooth_bars: int = 1,
    label_mode: str = "ma_trend",
    label_ma_window: int = 5,
) -> np.ndarray:
    """Retorna delta relativo de preco alinhado aos rotulos de classificacao."""
    _, masks = sequence_labels(
        prices,
        lookback,
        max(1, int(label_horizon_bars)),
        smooth_bars=max(1, int(label_smooth_bars)),
        label_mode=label_mode,
        ma_window=label_ma_window,
    )
    count = len(masks)
    if count == 0:
        return np.empty((0,), dtype=np.float32)
    horizon = max(1, int(label_horizon_bars))
    deltas = np.zeros(count, dtype=np.float32)
    for offset, end_idx in enumerate(range(lookback, lookback + count)):
        start_idx = end_idx - 1
        future_idx = min(len(prices) - 1, end_idx + horizon - 1)
        base = float(prices[start_idx])
        future = float(prices[future_idx])
        if abs(base) < 1e-12:
            deltas[offset] = 0.0
        else:
            deltas[offset] = (future - base) / abs(base)
    return deltas


def extract_features(prices: np.ndarray, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compatibilidade: retorna ultima linha de cada sequencia como matriz 2D."""
    seqs, targets, _ = extract_sequences(prices, lookback)
    if len(seqs) == 0:
        return np.empty((0, FEATURE_DIM)), np.empty((0,))
    flat = seqs[:, -1, :]
    return flat, targets
