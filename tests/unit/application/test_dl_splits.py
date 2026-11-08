"""Testes unitarios para purged_temporal_splits e splits_valid."""

from src.application.services.deep_learning.dl_splits import (
    purged_temporal_splits,
    splits_valid,
)


def test_splits_valid():
    assert splits_valid(20, 10, 30, 25) is True
    assert splits_valid(10, 10, 30, 25) is False
    assert splits_valid(20, 10, 25, 25) is False


def test_purged_temporal_splits_too_small():
    assert purged_temporal_splits(15, 5) is None


def test_purged_temporal_splits_sparse_samples_with_large_val_bars():
    splits_391 = purged_temporal_splits(391, 4998)
    assert splits_391 is not None
    train_sl, val_sl, calib_sl = splits_391
    assert train_sl.stop > 0
    assert val_sl.stop > val_sl.start
    assert calib_sl.stop > calib_sl.start

    splits_1400 = purged_temporal_splits(1400, 4998)
    assert splits_1400 is not None
    tr_1400, va_1400, ca_1400 = splits_1400
    assert tr_1400.stop > 700
    assert va_1400.stop > va_1400.start
    assert ca_1400.stop > ca_1400.start


def test_purged_temporal_splits_full_dataset():
    splits_full = purged_temporal_splits(25000, 4998)
    assert splits_full is not None
    tr_full, va_full, ca_full = splits_full
    assert tr_full.stop > 15000
    assert va_full.stop - va_full.start == 4998
