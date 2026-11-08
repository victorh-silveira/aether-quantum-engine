"""Extracao de sequencias TCN e rotulos binarios a partir de precos OHLC."""

import numpy as np

from src.application.services.deep_learning.dl_feature_build import (
    FEATURE_DIM,
    precompute_price_series,
)
from src.application.services.deep_learning.dl_feature_matrix import build_feature_matrix
from src.application.services.deep_learning.dl_labels import sequence_labels


def extract_sequences_and_deltas(
    prices: np.ndarray,
    lookback: int,
    *,
    granularity: int = 60,
    label_horizon_bars: int = 1,
    label_smooth_bars: int = 1,
    label_mode: str = "ma_trend",
    label_ma_window: int = 5,
    implied_vol_bars: int = 60,
    symbol: str = "1HZ75V",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
    filter_active: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extrai tensores (N, L, F), rotulos binarios, mascara ativa e deltas relativos."""
    n = len(prices)
    horizon = max(1, int(label_horizon_bars))
    smooth = max(1, int(label_smooth_bars))
    tail = horizon + smooth
    if n < lookback + tail:
        return (
            np.empty((0, lookback, FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
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
        open_=open_,
        high=high,
        low=low,
        series=series,
    )
    count = len(targets)
    if count == 0:
        return (
            np.empty((0, lookback, FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    sequences = []
    deltas = np.zeros(count, dtype=np.float32)
    has_open = open_ is not None and len(open_) == len(prices)
    for offset, i in enumerate(range(lookback, lookback + count)):
        sequences.append(feature_matrix[i - lookback + 1 : i + 1])
        future_idx = i + horizon
        if future_idx < len(prices):
            base = float(open_[future_idx]) if has_open else float(prices[i])
            future = float(np.mean(prices[future_idx : future_idx + smooth]))
            deltas[offset] = 0.0 if abs(base) < 1e-12 else (future - base) / abs(base)

    seq_arr = np.array(sequences, dtype=np.float32)
    if filter_active:
        active_idx = np.where(masks > 0.5)[0]
        if len(active_idx) >= 50:
            return seq_arr[active_idx], targets[active_idx], masks[active_idx], deltas[active_idx]
    return seq_arr, targets, masks, deltas


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
    symbol: str = "1HZ75V",
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    micro: dict[str, np.ndarray] | None = None,
    filter_active: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrai tensores (N, L, F), rotulos binarios e mascara ativa com filtro opcional."""
    seqs, targets, masks, _ = extract_sequences_and_deltas(
        prices,
        lookback,
        granularity=granularity,
        label_horizon_bars=label_horizon_bars,
        label_smooth_bars=label_smooth_bars,
        label_mode=label_mode,
        label_ma_window=label_ma_window,
        implied_vol_bars=implied_vol_bars,
        symbol=symbol,
        open_=open_,
        high=high,
        low=low,
        micro=micro,
        filter_active=filter_active,
    )
    return seqs, targets, masks


def sequence_price_deltas(
    prices: np.ndarray,
    lookback: int,
    *,
    label_horizon_bars: int = 1,
    label_smooth_bars: int = 1,
    label_mode: str = "ma_trend",
    label_ma_window: int = 5,
    open_: np.ndarray | None = None,
    filter_active: bool = False,
) -> np.ndarray:
    """Retorna delta relativo de preco alinhado aos rotulos de classificacao."""
    _, _, _, deltas = extract_sequences_and_deltas(
        prices,
        lookback,
        label_horizon_bars=label_horizon_bars,
        label_smooth_bars=label_smooth_bars,
        label_mode=label_mode,
        label_ma_window=label_ma_window,
        open_=open_,
        filter_active=filter_active,
    )
    return deltas


def extract_features(prices: np.ndarray, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compatibilidade: retorna ultima linha de cada sequencia como matriz 2D."""
    seqs, targets, _ = extract_sequences(prices, lookback)
    if len(seqs) == 0:
        return np.empty((0, FEATURE_DIM)), np.empty((0,))
    flat = seqs[:, -1, :]
    return flat, targets
