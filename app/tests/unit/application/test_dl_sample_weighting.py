"""Testes de sample weighting DL (classe + recencia)."""

import numpy as np

from src.application.services.deep_learning.dl_sample_weighting import (
    align_sample_weights,
    apply_class_balance_weights,
    apply_recency_half_life,
    compose_train_weights,
    label_call_fraction,
    minority_class_recall,
    parse_sample_weighting_config,
)


def test_label_call_fraction_empty_and_skewed():
    assert label_call_fraction([]) == 0.5
    assert label_call_fraction([1.0, 1.0, 0.0, 1.0]) == 0.75


def test_class_balance_weights_boosts_minority():
    y = [1.0] * 8 + [0.0] * 2
    base = [1.0] * 10
    out = apply_class_balance_weights(base, y, enabled=True, imbalance_eps=0.05)
    assert out[0] < out[-1]


def test_class_balance_disabled_passthrough():
    y = [1.0] * 9 + [0.0]
    base = [2.0] * 10
    assert apply_class_balance_weights(base, y, enabled=False) == base


def test_recency_half_life_favors_tail():
    weights = [1.0] * 8
    out = apply_recency_half_life(weights, half_life_n=2, enabled=True)
    assert out[-1] > out[0]


def test_align_sample_weights_full_and_train_len():
    full = [float(i) for i in range(10)]
    aligned = align_sample_weights(full, full_n=10, train_index=slice(0, 6))
    assert aligned == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert align_sample_weights([9.0, 8.0], full_n=10, train_index=slice(0, 2)) == [9.0, 8.0]


def test_compose_train_weights_pipeline():
    y = np.asarray([1.0] * 7 + [0.0] * 3, dtype=np.float32)
    cfg = {
        "class_balance_enabled": True,
        "class_balance_eps": 0.05,
        "recency_enabled": True,
        "recency_half_life_n": 4,
    }
    out = compose_train_weights([1.0] * 10, y, full_n=10, train_index=slice(0, 10), weighting_cfg=cfg)
    assert len(out) == 10
    assert out[-1] > 0.0


def test_minority_class_recall_put_minority():
    y = [1.0] * 8 + [0.0] * 2
    pred = [True] * 9 + [False]
    assert minority_class_recall(y, pred) == 0.5


def test_parse_sample_weighting_from_ssot():
    cfg = parse_sample_weighting_config({})
    assert cfg["class_balance_enabled"] is True
    assert cfg["recency_half_life_n"] >= 1


def test_class_balance_empty_and_mismatch_and_balanced():
    assert apply_class_balance_weights([], [], enabled=True) == []
    assert apply_class_balance_weights([1.0, 1.0], [1.0], enabled=True) == [1.0, 1.0]
    balanced = [1.0, 0.0, 1.0, 0.0]
    assert apply_class_balance_weights([1.0] * 4, balanced, enabled=True, imbalance_eps=0.05) == [1.0] * 4


def test_recency_disabled_and_empty():
    assert apply_recency_half_life([], half_life_n=2, enabled=True) == []
    assert apply_recency_half_life([1.0, 2.0], half_life_n=2, enabled=False) == [1.0, 2.0]


def test_align_sample_weights_edge_cases():
    assert align_sample_weights(None, full_n=0, train_index=slice(0, 0)) == []
    assert align_sample_weights(None, full_n=4, train_index=slice(0, 3)) == [1.0, 1.0, 1.0]
    assert align_sample_weights([3.0], full_n=4, train_index=slice(0, 2)) == [1.0, 1.0]


def test_minority_recall_edges():
    assert minority_class_recall([], []) == 1.0
    assert minority_class_recall([1.0, 1.0], [True]) == 1.0
    assert minority_class_recall([1.0, 1.0], [True, True]) == 0.0
    y = [0.0] * 8 + [1.0] * 2
    pred = [False] * 8 + [True, False]
    assert minority_class_recall(y, pred) == 0.5
