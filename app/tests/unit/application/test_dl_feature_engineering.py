import numpy as np
import pytest

from src.application.services.deep_learning.dl_feature_build import (
    FEATURE_DIM,
    attach_microstructure,
    precompute_price_series,
    symbol_vol_target,
)
from src.application.services.deep_learning.dl_feature_indicators import (
    calculate_adx,
    calculate_cmo,
    calculate_ema_crossover,
    calculate_keltner_channel_pct_b,
    calculate_stochastic,
    calculate_volatility_ratio,
    calculate_williams_r,
)
from src.application.services.deep_learning.dl_feature_matrix import (
    build_feature_matrix,
    build_feature_row,
    build_sequence_tensor,
)
from src.application.services.deep_learning.dl_feature_normalize import causal_robust_scale
from src.application.services.deep_learning.dl_feature_orthogonal import (
    ORTHOGONAL_FEATURE_NAMES,
    _hurst_centered,
)


def test_feature_dim_is_fourteen():
    assert FEATURE_DIM == 14
    assert len(ORTHOGONAL_FEATURE_NAMES) == 14


def test_log_return_and_ema_distances():
    prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.0, 105.0], dtype=np.float64)
    series = precompute_price_series(prices, granularity=60, symbol="R_10")
    assert "log_return" in series
    assert "ema_dist_20" in series
    assert "ema_dist_50" in series
    assert "ema_20_50_dist" in series
    assert "bb_width_raw" in series
    assert "atr_raw" in series
    assert "delta_rsi" in series
    assert "macd" in series
    assert "stoch_k" in series
    assert "adx" in series
    assert np.isclose(series["log_return"][1], np.log(101.0 / 100.0), rtol=1e-5)
    assert np.isfinite(series["ema_dist_50"]).all()


def test_build_feature_row_shape():
    prices = np.linspace(100.0, 110.0, 80)
    series = precompute_price_series(prices, granularity=60, symbol="R_10")
    row = build_feature_row(series, 40)
    assert row.shape == (FEATURE_DIM,)
    matrix = build_feature_matrix(series)
    assert matrix.shape == (len(prices), FEATURE_DIM)


def test_causal_norm_no_lookahead():
    rng = np.random.default_rng(0)
    x = rng.normal(size=64)
    scaled = causal_robust_scale(x, window=16, min_hist=4)
    assert scaled.shape == x.shape
    assert np.isfinite(scaled).all()
    scaled_prefix = causal_robust_scale(x[:40], window=16, min_hist=4)
    assert np.allclose(scaled[:40], scaled_prefix)


def test_hurst_centered_changes_last_feature():
    prices = np.linspace(100.0, 120.0, 60)
    series = precompute_price_series(prices, granularity=60, symbol="R_10")
    row = build_feature_row(series, 50)
    assert row.shape == (14,)
    assert np.isfinite(row[-1])


def test_precompute_attaches_macro_closes():
    prices = np.linspace(100.0, 101.0, 16)
    macro = np.linspace(200.0, 201.0, 16)
    series = precompute_price_series(prices, granularity=60, symbol="R_10", macro_closes=macro)
    assert np.allclose(series["macro_closes"], macro)


def test_hurst_centered_pads_short_series():
    padded = _hurst_centered({"hurst": np.array([0.6, 0.7])}, 4)
    assert padded.shape == (4,)
    assert padded[0] == pytest.approx(0.1)
    assert padded[2] == pytest.approx(0.0)
    aligned = _hurst_centered({"hurst": np.array([0.4, 0.5, 0.6])}, 3)
    assert aligned.tolist() == pytest.approx([-0.1, 0.0, 0.1])


def test_calculate_stochastic_flat_prices():
    prices = np.full(50, 100.0)
    k, d = calculate_stochastic(prices, prices, prices, period=14, smooth_k=3)
    assert (k == 0.5).all()


def test_new_indicators_edge_cases():
    prices = np.array([100.0, 100.0])
    adx, di_diff = calculate_adx(prices, prices, prices, period=14)
    assert (adx == 0.0).all()
    assert (di_diff == 0.0).all()
    wr = calculate_williams_r(prices, prices, prices, period=14)
    assert (wr == 0.5).all()
    prices = np.full(50, 100.0)
    dist = calculate_ema_crossover(prices, fast=9, slow=21)
    assert (dist == 0.0).all()
    cmo = calculate_cmo(prices, period=14)
    assert np.isfinite(cmo).all()
    kelt = calculate_keltner_channel_pct_b(prices, prices, prices, period=20, atr_period=10, atr_mult=1.5)
    assert np.isfinite(kelt).all()
    vr = calculate_volatility_ratio(np.zeros(50), short=5, long=20)
    assert np.isfinite(vr).all()


def test_attach_microstructure_and_symbol_vol():
    prices = np.linspace(100.0, 105.0, 20)
    series = precompute_price_series(prices, granularity=60, symbol="1HZ75V")
    attach_microstructure(series, None)
    assert "tick_count" in series
    assert symbol_vol_target("1HZ75V") == 0.75


def test_build_sequence_tensor_shape():
    prices = np.linspace(100.0, 110.0, 40)
    seq = build_sequence_tensor(prices, lookback=10, end_index=30, granularity=60, symbol="R_10")
    assert seq.shape == (10, FEATURE_DIM)


def test_symbol_vol_target_branch_table():
    assert symbol_vol_target("R_25") == 0.25
    assert symbol_vol_target("1HZ25V") == 0.25
    assert symbol_vol_target("R_100") == 1.00
    assert symbol_vol_target("1HZ100V") == 1.00


def test_build_feature_row_matrix_override_knobs():
    prices = np.linspace(100.0, 110.0, 60)
    series = precompute_price_series(prices, granularity=60, symbol="R_10")
    row = build_feature_row(series, 30, causal_norm_window=16, causal_norm_clip=2.5)
    assert row.shape == (FEATURE_DIM,)
    matrix = build_feature_matrix(series, causal_norm_window=16, causal_norm_clip=2.5)
    assert matrix.shape == (len(prices), FEATURE_DIM)
