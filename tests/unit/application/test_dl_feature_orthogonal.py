"""Testes unitarios para matriz de features ortogonais 14D e pre-flight benchmark gate."""

import numpy as np

from src.application.services.deep_learning.dl_feature_build import (
    FEATURE_DIM,
    MICRO_FEATURE_DIM,
    precompute_price_series,
)
from src.application.services.deep_learning.dl_feature_matrix import (
    build_sequence_tensor,
)
from src.application.services.deep_learning.dl_feature_orthogonal import (
    FEATURE_DIM as ORTH_DIM,
    ORTHOGONAL_FEATURE_NAMES,
    UNBOUNDED_COLS,
    build_orthogonal_feature_matrix,
    build_orthogonal_feature_row,
    build_orthogonal_raw_matrix,
)
from src.application.services.deep_learning.dl_preflight_benchmark import run_preflight_linear_baseline


def test_orthogonal_feature_constants():
    """Valida dimensoes e consistencia das constantes de features 14D limpas."""
    assert FEATURE_DIM == 14
    assert ORTH_DIM == 14
    assert MICRO_FEATURE_DIM == 0
    assert len(ORTHOGONAL_FEATURE_NAMES) == 14
    assert "norm_log_ret_1" in ORTHOGONAL_FEATURE_NAMES
    assert "norm_frac_diff" in ORTHOGONAL_FEATURE_NAMES
    assert "buy_tick_ratio_centered" not in ORTHOGONAL_FEATURE_NAMES
    assert UNBOUNDED_COLS == (0, 1, 3, 5, 6, 7, 8, 9, 12, 13)


def test_clean_microstructure_defaults():
    """Valida que o batch sem ticks utiliza estritamente defaults neutros."""
    n = 50
    prices = np.linspace(100.0, 110.0, n)
    high = prices + 0.5
    low = prices - 0.5
    open_ = prices - 0.1
    series = precompute_price_series(
        prices,
        granularity=60,
        symbol="1HZ75V",
        open_=open_,
        high=high,
        low=low,
        micro=None,
    )
    assert "buy_tick_ratio" in series
    assert len(series["buy_tick_ratio"]) == n
    assert np.allclose(series["buy_tick_ratio"], 0.5)


def test_build_orthogonal_raw_and_scaled_matrix_shape():
    """Valida shape 14D da matriz bruta e escalonada."""
    n = 60
    prices = np.linspace(100.0, 105.0, n)
    high = prices + 0.3
    low = prices - 0.3
    open_ = prices - 0.05
    series = precompute_price_series(
        prices,
        granularity=60,
        symbol="1HZ75V",
        open_=open_,
        high=high,
        low=low,
    )
    raw = build_orthogonal_raw_matrix(series)
    assert raw.shape == (n, 14)
    assert raw.dtype == np.float32

    scaled = build_orthogonal_feature_matrix(series)
    assert scaled.shape == (n, 14)
    assert scaled.dtype == np.float32
    assert not np.isnan(scaled).any()
    assert not np.isinf(scaled).any()


def test_build_orthogonal_feature_row_and_sequence_tensor():
    """Valida vetor de features por linha e tensor sequencial 14D."""
    n = 40
    prices = np.linspace(100.0, 102.0, n)
    high = prices + 0.2
    low = prices - 0.2
    open_ = prices - 0.02
    series = precompute_price_series(
        prices,
        granularity=60,
        symbol="1HZ75V",
        open_=open_,
        high=high,
        low=low,
    )
    row = build_orthogonal_feature_row(series, index=n - 1)
    assert row.shape == (14,)
    assert row.dtype == np.float32

    seq = build_sequence_tensor(
        prices,
        lookback=8,
        end_index=n - 1,
        granularity=60,
        symbol="1HZ75V",
        open_=open_,
        high=high,
        low=low,
    )
    assert seq.shape == (8, 14)
    assert seq.dtype == np.float32


def test_preflight_linear_baseline_evaluation():
    """Valida computo do benchmark linear pre-flight."""
    np.random.seed(42)
    x_tr = np.random.randn(100, 8, 14)
    y_tr = np.random.randint(0, 2, size=100)
    x_val = np.random.randn(30, 8, 14)
    y_val = np.random.randint(0, 2, size=30)
    acc_dummy, acc_linear = run_preflight_linear_baseline(x_tr, y_tr, x_val, y_val)
    assert 0.0 <= acc_dummy <= 1.0
    assert 0.0 <= acc_linear <= 1.0
