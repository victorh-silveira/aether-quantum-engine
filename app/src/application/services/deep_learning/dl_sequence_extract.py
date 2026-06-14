"""Extracao de sequencias TCN e rotulos binarios a partir de precos OHLC."""

import numpy as np

from src.application.services.deep_learning.dl_feature_build import (
    FEATURE_DIM,
    build_feature_matrix,
    precompute_price_series,
)
from src.application.services.deep_learning.dl_labels import sequence_labels


def extract_sequences(
    prices: np.ndarray,
    lookback: int,
    *,
    granularity: int = 60,
    label_horizon_bars: int = 1,
    symbol: str = "R_50",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrai tensores (N, L, F), rotulos binarios e mascara ativa."""
    n = len(prices)
    horizon = max(1, int(label_horizon_bars))
    if n < lookback + horizon + 1:
        return np.empty((0, lookback, FEATURE_DIM)), np.empty((0,)), np.empty((0,))
    series = precompute_price_series(
        prices,
        granularity=granularity,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
    )
    feature_matrix = build_feature_matrix(series)
    targets, masks = sequence_labels(prices, lookback, horizon)
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


def extract_features(prices: np.ndarray, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compatibilidade: retorna ultima linha de cada sequencia como matriz 2D."""
    seqs, targets, _ = extract_sequences(prices, lookback)
    if len(seqs) == 0:
        return np.empty((0, FEATURE_DIM)), np.empty((0,))
    flat = seqs[:, -1, :]
    return flat, targets
