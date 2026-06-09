"""Extracao de sequencias TCN e meta-labels a partir de precos OHLC."""

import numpy as np

from src.application.services.deep_learning.dl_feature_build import (
    FEATURE_DIM,
    build_sequence_tensor,
)
from src.application.services.deep_learning.dl_pair_features import spread_confirms_direction


def extract_sequences(
    prices: np.ndarray,
    lookback: int,
    *,
    label_min_move_pct: float = 0.0002,
    granularity: int = 300,
    pair_prices: np.ndarray | None = None,
    require_pair_label: bool = False,
    sym_is_bull: bool = True,
    label_horizon_bars: int = 1,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrai tensores (N, L, F), rotulos binarios e mascara meta-label."""
    n = len(prices)
    horizon = max(1, int(label_horizon_bars))
    min_tail = horizon + 2
    if n < lookback + min_tail:
        return np.empty((0, lookback, FEATURE_DIM)), np.empty((0,)), np.empty((0,))
    threshold = max(0.0, float(label_min_move_pct))
    sequences = []
    targets = []
    masks = []
    last_i = n - horizon - 1
    for i in range(lookback, last_i + 1):
        j = i + horizon
        move = abs(prices[j] - prices[i]) / (prices[i] + 1e-10)
        target_up = prices[j] > prices[i]
        pair_ok = True
        if require_pair_label and pair_prices is not None and len(pair_prices) >= n:
            pair_ok = spread_confirms_direction(
                prices,
                pair_prices,
                i,
                target_up=target_up,
                sym_is_bull=sym_is_bull,
                horizon_bars=horizon,
            )
        active = move >= threshold and pair_ok
        sequences.append(
            build_sequence_tensor(
                prices,
                lookback,
                i,
                granularity=granularity,
                pair_prices=pair_prices,
                open_=open_,
                high=high,
                low=low,
            )
        )
        targets.append(1.0 if target_up else 0.0)
        masks.append(1.0 if active else 0.0)
    return (
        np.array(sequences, dtype=np.float32),
        np.array(targets, dtype=np.float32),
        np.array(masks, dtype=np.float32),
    )


def extract_features(prices: np.ndarray, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compatibilidade: retorna ultima linha de cada sequencia como matriz 2D."""
    seqs, targets, _ = extract_sequences(prices, lookback)
    if len(seqs) == 0:
        return np.empty((0, FEATURE_DIM)), np.empty((0,))
    flat = seqs[:, -1, :]
    return flat, targets
