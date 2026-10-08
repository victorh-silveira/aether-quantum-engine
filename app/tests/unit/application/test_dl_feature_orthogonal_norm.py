import numpy as np

from src.application.services.deep_learning.dl_feature_normalize import (
    apply_causal_column_scale,
    causal_robust_scale,
    center_unit_interval,
)
from src.application.services.deep_learning.dl_feature_orthogonal import (
    FEATURE_DIM,
    build_orthogonal_feature_matrix,
    build_orthogonal_raw_matrix,
)


def test_center_and_causal_scale_edges():
    assert center_unit_interval(np.array([0.0, 0.5, 1.0])).tolist() == [-1.0, 0.0, 1.0]
    empty = causal_robust_scale(np.array([], dtype=np.float64))
    assert empty.shape == (0,)
    short = causal_robust_scale(np.array([1.0, 2.0]), window=10, min_hist=8)
    assert short.tolist() == [0.0, 0.0]
    mat = apply_causal_column_scale(np.zeros((0, 2), dtype=np.float32), (0,))
    assert mat.shape == (0, 2)
    mat2 = apply_causal_column_scale(np.ones((5, 2), dtype=np.float32), (-1, 9, 0))
    assert mat2.shape == (5, 2)


def test_orthogonal_without_macro():
    n = 40
    series = {
        "log_return": np.linspace(-0.01, 0.01, n),
        "rsi": np.full(n, 0.55),
        "bb_pct_b": np.full(n, 0.4),
        "bb_width_raw": np.full(n, 0.02),
        "bb_width": np.full(n, 0.02),
        "atr_raw": np.linspace(0.1, 0.2, n),
        "atr_norm": np.linspace(0.1, 0.2, n),
        "macd": np.linspace(-0.01, 0.01, n),
        "macd_signal": np.zeros(n),
        "stoch_k": np.full(n, 0.6),
        "adx": np.full(n, 0.2),
        "delta_rsi": np.zeros(n),
        "ema_9_21_dist": np.linspace(-0.01, 0.01, n),
        "ema_20_50_dist": np.linspace(-0.02, 0.02, n),
        "ema_dist_50": np.linspace(-0.02, 0.02, n),
        "vol_ratio_short_long": np.full(n, 1.1),
    }
    raw = build_orthogonal_raw_matrix(series)
    assert raw.shape == (n, FEATURE_DIM)
    matrix = build_orthogonal_feature_matrix(series, causal_norm_window=16)
    assert matrix.shape == (n, FEATURE_DIM)
