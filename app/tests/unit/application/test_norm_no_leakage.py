import numpy as np

from src.application.services.deep_learning.model import fit_norm_stats, normalize_sequences


def test_norm_fit_only_on_train_slice():
    x = np.random.default_rng(0).normal(size=(100, 48, 18)).astype(np.float32)
    train_sl = slice(0, 70)
    val_sl = slice(70, 100)
    stats = fit_norm_stats(x[train_sl])
    train_norm = normalize_sequences(x[train_sl], stats)
    val_norm = normalize_sequences(x[val_sl], stats)
    assert train_norm.shape == x[train_sl].shape
    assert val_norm.shape == x[val_sl].shape
    assert np.isfinite(val_norm).all()
