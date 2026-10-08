import numpy as np

from src.application.services.deep_learning.dl_sequence_extract import (
    extract_features,
    extract_sequences,
    extract_sequences_and_deltas,
    sequence_price_deltas,
)


def test_extract_sequences_empty_and_short():
    prices_short = np.array([100.0, 101.0], dtype=np.float64)
    seqs, targets, masks = extract_sequences(prices_short, lookback=10)
    assert len(seqs) == 0
    assert len(targets) == 0
    assert len(masks) == 0

    flat, targets_f = extract_features(prices_short, lookback=10)
    assert len(flat) == 0
    assert len(targets_f) == 0


def test_extract_sequences_and_deltas_regular():
    prices = np.linspace(100.0, 120.0, 50, dtype=np.float64)
    seqs, targets, masks, deltas = extract_sequences_and_deltas(
        prices, lookback=10, label_horizon_bars=1, label_smooth_bars=1
    )
    assert len(seqs) == len(targets) == len(masks) == len(deltas)
    assert len(seqs) > 0
    assert seqs.ndim == 3
    assert seqs.shape[1] == 10

    flat, targets_feat = extract_features(prices, lookback=10)
    assert len(flat) == len(targets_feat)
    assert flat.shape == (len(seqs), seqs.shape[2])


def test_extract_sequences_filter_active():
    prices = np.linspace(100.0, 150.0, 100, dtype=np.float64)
    seqs_all, targets_all, masks_all = extract_sequences(prices, lookback=10, label_horizon_bars=1, filter_active=False)
    seqs_act, targets_act, masks_act = extract_sequences(prices, lookback=10, label_horizon_bars=1, filter_active=True)
    assert len(seqs_all) >= len(seqs_act)
    assert len(seqs_act) == len(targets_act) == len(masks_act)


def test_sequence_price_deltas_direct():
    prices = np.array([100.0, 102.0, 104.0, 106.0, 108.0] * 10, dtype=np.float64)
    open_ = prices.copy()
    deltas = sequence_price_deltas(prices, lookback=10, label_horizon_bars=1, label_smooth_bars=1, open_=open_)
    assert len(deltas) > 0
    assert isinstance(deltas, np.ndarray)

    deltas_short = sequence_price_deltas(np.array([100.0]), lookback=5)
    assert len(deltas_short) == 0
