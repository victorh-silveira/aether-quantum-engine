import numpy as np

from src.application.services.deep_learning.dl_barrier_labels import (
    quantum_multi_barrier_direction,
    quantum_multi_barrier_label_and_mask,
    triple_barrier_direction,
    triple_barrier_label_and_mask,
)


def test_triple_barrier_with_series_regime_filtering():
    prices = np.array([100.0] * 20 + [100.0, 105.0])
    series_neutral = {
        "bb_pct_b": np.array([0.5] * 22),
        "keltner_pct_b": np.array([0.5] * 22),
        "vol_ratio_short_long": np.array([1.0] * 22),
        "rsi": np.array([0.5] * 22),
        "delta_rsi": np.array([0.01] * 22),
        "adx": np.array([0.15] * 22),
        "di_diff": np.array([0.02] * 22),
        "atr_raw": np.array([0.01] * 22),
    }
    up, mask = triple_barrier_label_and_mask(prices, 20, 1, series=series_neutral)
    assert mask == 0.0

    series_overbought = {
        "bb_pct_b": np.array([0.98] * 22),
        "keltner_pct_b": np.array([0.99] * 22),
        "vol_ratio_short_long": np.array([1.30] * 22),
        "rsi": np.array([0.85] * 22),
        "delta_rsi": np.array([0.06] * 22),
        "adx": np.array([0.30] * 22),
        "di_diff": np.array([0.15] * 22),
        "atr_raw": np.array([0.01] * 22),
    }
    prices_drop = np.array([100.0] * 20 + [100.0, 95.0])
    up_act, mask_act = triple_barrier_label_and_mask(prices_drop, 20, 1, series=series_overbought)
    assert up_act is False
    assert mask_act == 1.0

    prices_rise = np.array([100.0] * 20 + [100.0, 105.0])
    up_fail, mask_fail = triple_barrier_label_and_mask(prices_rise, 20, 1, series=series_overbought)
    assert up_fail is True
    assert mask_fail == 1.0

    series_oversold = {
        "bb_pct_b": np.array([0.02] * 22),
        "rsi": np.array([0.15] * 22),
        "delta_rsi": np.array([-0.06] * 22),
        "atr_raw": np.array([0.01] * 22),
    }
    up_ov, mask_ov = triple_barrier_label_and_mask(prices_rise, 20, 1, series=series_oversold)
    assert up_ov is True
    assert mask_ov == 1.0


def test_triple_barrier_high_low_touches():
    prices = np.array([100.0] * 20 + [100.0, 101.0])
    high = np.array([100.0] * 20 + [100.0, 110.0])
    low = np.array([100.0] * 20 + [100.0, 99.0])
    up_h, mask_h = triple_barrier_label_and_mask(prices, 20, 1, lookback_vol=5, barrier_mult=1.0, high=high, low=low)
    assert up_h is True
    assert mask_h == 1.0

    prices_l = np.array([100.0] * 20 + [100.0, 99.0])
    high_l = np.array([100.0] * 20 + [100.0, 100.0])
    low_l = np.array([100.0] * 20 + [100.0, 90.0])
    up_l, mask_l = triple_barrier_label_and_mask(
        prices_l, 20, 1, lookback_vol=5, barrier_mult=1.0, high=high_l, low=low_l
    )
    assert up_l is False
    assert mask_l == 1.0


def test_quantum_multi_barrier_min_viable_delta_and_series():
    prices = np.array([100.0] * 20 + [100.0, 105.0])
    series_neutral = {
        "bb_pct_b": np.array([0.5] * 22),
        "keltner_pct_b": np.array([0.5] * 22),
        "vol_ratio_short_long": np.array([1.0] * 22),
        "rsi": np.array([0.5] * 22),
        "delta_rsi": np.array([0.01] * 22),
        "adx": np.array([0.15] * 22),
        "di_diff": np.array([0.02] * 22),
        "atr_raw": np.array([0.01] * 22),
    }
    up, mask = quantum_multi_barrier_label_and_mask(prices, 20, 1, series=series_neutral)
    assert mask == 0.0

    up_m1, mask_m1 = quantum_multi_barrier_label_and_mask(prices, 20, 1, lookback_vol=5, min_viable_delta=0.01)
    assert up_m1 is True
    assert mask_m1 == 1.0

    prices_drop = np.array([100.0] * 20 + [100.0, 95.0])
    up_m2, mask_m2 = quantum_multi_barrier_label_and_mask(prices_drop, 20, 1, lookback_vol=5, min_viable_delta=0.01)
    assert up_m2 is False
    assert mask_m2 == 1.0

    prices_flat = np.array([100.0] * 20 + [100.0, 100.05])
    _, mask_m3 = quantum_multi_barrier_label_and_mask(prices_flat, 20, 1, lookback_vol=5, min_viable_delta=0.01)
    assert mask_m3 == 0.0


def test_barrier_direction_functions():
    prices = np.array([100.0] * 20 + [100.0, 110.0])
    assert triple_barrier_direction(prices, 20, 1) is True
    assert quantum_multi_barrier_direction(prices, 20, 1) is True


def test_triple_barrier_price_direct_touches_and_dead_zone():
    prices_up = np.array([100.0] * 20 + [100.0, 115.0])
    up_t, mask_t = triple_barrier_label_and_mask(prices_up, 20, 1, lookback_vol=5, barrier_mult=1.0)
    assert up_t is True
    assert mask_t == 1.0

    prices_down = np.array([100.0] * 20 + [100.0, 85.0])
    up_b, mask_b = triple_barrier_label_and_mask(prices_down, 20, 1, lookback_vol=5, barrier_mult=1.0)
    assert up_b is False
    assert mask_b == 1.0

    prices_dead = np.array([100.0] * 20 + [100.0, 100.0001])
    up_d, mask_d = triple_barrier_label_and_mask(prices_dead, 20, 1, lookback_vol=5, barrier_mult=1.0)
    assert mask_d == 0.0


def test_quantum_multi_barrier_direct_touches():
    prices_up = np.array([100.0] * 20 + [100.0, 115.0])
    up_q, mask_q = quantum_multi_barrier_label_and_mask(prices_up, 20, 1, lookback_vol=5)
    assert up_q is True
    assert mask_q == 1.0

    prices_down = np.array([100.0] * 20 + [100.0, 85.0])
    up_qd, mask_qd = quantum_multi_barrier_label_and_mask(prices_down, 20, 1, lookback_vol=5)
    assert up_qd is False
    assert mask_qd == 1.0


def test_barrier_close_decides_when_candle_crosses_both_sides():
    high = np.array([100.0] * 20 + [100.0, 110.0])
    low = np.array([100.0] * 20 + [100.0, 90.0])
    prices_up = np.array([100.0] * 20 + [100.0, 105.0])
    prices_down = np.array([100.0] * 20 + [100.0, 95.0])
    assert triple_barrier_label_and_mask(prices_up, 20, 1, high=high, low=low)[0] is True
    assert triple_barrier_label_and_mask(prices_down, 20, 1, high=high, low=low)[0] is False
    assert quantum_multi_barrier_label_and_mask(prices_up, 20, 1, high=high, low=low)[0] is True
    assert quantum_multi_barrier_label_and_mask(prices_down, 20, 1, high=high, low=low)[0] is False
