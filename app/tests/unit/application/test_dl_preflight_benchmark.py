import logging

import numpy as np

from src.application.services.deep_learning.dl_preflight_benchmark import (
    run_preflight_linear_baseline,
)


def test_preflight_linear_baseline_insufficient():
    x_tr = np.zeros((10, 8, 14), dtype=np.float32)
    y_tr = np.zeros(10, dtype=np.float32)
    x_val = np.zeros((2, 8, 14), dtype=np.float32)
    y_val = np.zeros(2, dtype=np.float32)
    acc_dummy, acc_linear = run_preflight_linear_baseline(x_tr, y_tr, x_val, y_val)
    assert acc_dummy == 0.5
    assert acc_linear == 0.5


def test_preflight_linear_baseline_single_class():
    x_tr = np.random.randn(30, 8, 14).astype(np.float32)
    y_tr = np.ones(30, dtype=np.float32)
    x_val = np.random.randn(10, 8, 14).astype(np.float32)
    y_val = np.ones(10, dtype=np.float32)
    acc_dummy, acc_linear = run_preflight_linear_baseline(x_tr, y_tr, x_val, y_val)
    assert acc_dummy == 0.5
    assert acc_linear == 0.5


def test_preflight_linear_baseline_separable(caplog):
    np.random.seed(42)
    n_tr, n_val = 100, 40
    x_tr = np.random.randn(n_tr, 8, 14).astype(np.float32)
    x_tr[:, -1, 0] = np.where(x_tr[:, -1, 0] > 0.0, 5.0, -5.0)
    y_tr = (x_tr[:, -1, 0] > 0.0).astype(np.float32)

    x_val = np.random.randn(n_val, 8, 14).astype(np.float32)
    x_val[:, -1, 0] = np.where(x_val[:, -1, 0] > 0.0, 5.0, -5.0)
    y_val = (x_val[:, -1, 0] > 0.0).astype(np.float32)

    with caplog.at_level(logging.INFO, logger="AETH"):
        acc_dummy, acc_linear = run_preflight_linear_baseline(x_tr, y_tr, x_val, y_val)
    assert acc_linear > 0.80
    assert 0.0 <= acc_dummy <= 1.0
    majority_pct = max(float(np.mean(y_val)), 1.0 - float(np.mean(y_val))) * 100.0
    assert f"Maioria val: {majority_pct:.1f}%" in caplog.text
