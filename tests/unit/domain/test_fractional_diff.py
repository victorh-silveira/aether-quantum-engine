"""Testes unitarios para get_fractional_weights e frac_diff_causal."""

import numpy as np

from src.domain.math.fractional_diff import (
    frac_diff_causal,
    get_fractional_weights,
)


def test_get_fractional_weights_edge_cases():
    empty_w = get_fractional_weights(0.45, 0)
    assert len(empty_w) == 0

    zero_d_w = get_fractional_weights(0.0, 5)
    assert len(zero_d_w) == 5
    assert zero_d_w[0] == 1.0
    assert np.all(zero_d_w[1:] == 0.0)

    one_d_w = get_fractional_weights(1.0, 10, threshold=1e-4)
    assert len(one_d_w) == 2
    assert one_d_w[0] == 1.0
    assert one_d_w[1] == -1.0


def test_get_fractional_weights_fractional():
    w = get_fractional_weights(0.45, 100, threshold=1e-4)
    assert len(w) > 2
    assert w[0] == 1.0
    assert w[1] < 0.0
    assert np.all(np.abs(w) >= 1e-4)


def test_frac_diff_causal_identity_and_empty():
    empty_arr = np.empty(0, dtype=np.float64)
    res_empty = frac_diff_causal(empty_arr, d=0.45)
    assert len(res_empty) == 0

    series = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    res_zero = frac_diff_causal(series, d=0.0)
    assert np.allclose(res_zero, series)


def test_frac_diff_causal_shape_and_causality():
    series_a = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 104.0])
    series_b = np.copy(series_a)
    series_b[-1] = 999.0

    res_a = frac_diff_causal(series_a, d=0.45)
    res_b = frac_diff_causal(series_b, d=0.45)

    assert len(res_a) == len(series_a)
    assert np.allclose(res_a[:-1], res_b[:-1])
    assert not np.isclose(res_a[-1], res_b[-1])


def test_frac_diff_causal_2d_matrix():
    matrix = np.array(
        [
            [10.0, 20.0],
            [11.0, 22.0],
            [12.0, 21.0],
            [13.0, 23.0],
        ]
    )
    res_2d = frac_diff_causal(matrix, d=0.45)
    assert res_2d.shape == matrix.shape
    col0_single = frac_diff_causal(matrix[:, 0], d=0.45)
    assert np.allclose(res_2d[:, 0], col0_single)


def test_frac_diff_causal_with_nan():
    series = np.array([10.0, np.nan, 12.0, 13.0])
    res = frac_diff_causal(series, d=0.45)
    assert len(res) == 4
    assert np.isnan(res[1])
    assert not np.isnan(res[0])
