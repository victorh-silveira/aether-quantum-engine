import numpy as np

from src.application.services.deep_learning.dl_features import (
    FEATURE_DIM,
    build_feature_row,
    build_sequence_tensor,
    extract_sequences,
    precompute_price_series,
)
from src.application.services.deep_learning.dl_pair_features import precompute_pair_series


def test_build_feature_row_with_pair():
    prices = np.linspace(10.0, 20.0, 50)
    pair = np.linspace(5.0, 6.0, 50)
    series = precompute_price_series(prices, granularity=300)
    pair_series = precompute_pair_series(prices, pair)
    row = build_feature_row(series, 30, pair_series=pair_series)
    assert row.shape == (FEATURE_DIM,)


def test_extract_sequences_pair_label_filter():
    bull = np.linspace(100.0, 120.0, 80)
    bear = np.linspace(50.0, 45.0, 80)
    seqs, targets, masks = extract_sequences(
        bull,
        20,
        label_min_move_pct=0.0,
        pair_prices=bear,
        require_pair_label=True,
        granularity=300,
        sym_is_bull=True,
    )
    assert seqs.shape[1:] == (20, FEATURE_DIM)
    assert masks.sum() <= len(masks)


def test_build_sequence_tensor_pair():
    prices = np.sin(np.linspace(0, 5, 60)) + 10.0
    pair = np.cos(np.linspace(0, 5, 60)) + 5.0
    tensor = build_sequence_tensor(prices, 15, 40, granularity=300, pair_prices=pair)
    assert tensor.shape == (15, FEATURE_DIM)
