import pytest

from src.application.services import meta_classifier_features
from src.application.services.meta_classifier_features import (
    META_FEATURE_DIM,
    _base_feature_vector,
    extract_meta_feature_vector,
    meta_classifier_column_names,
)


def test_meta_classifier_column_names_include_cross_symbol_keys():
    names = meta_classifier_column_names()
    assert len(names) == META_FEATURE_DIM
    assert names[-5:] == [
        "cross_symbol_prob_delta",
        "cross_symbol_vol_ratio_diff",
        "cross_symbol_rsi_spread",
        "micro_tick_acceleration",
        "keltner_deviation_ratio",
    ]


def test_extract_meta_feature_vector_from_indicators_fallback():
    metrics = {
        "indicators": {"hurst": 0.42, "adx": 18.0},
        "calibrated_prob": 0.61,
        "val_accuracy": 0.7,
        "edge": 0.08,
    }
    vector = extract_meta_feature_vector(metrics)
    assert len(vector) == META_FEATURE_DIM
    assert vector[0] == pytest.approx(0.42)
    assert vector[-5:] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])


def test_base_feature_vector_truncates_when_values_exceed_feature_dim(monkeypatch):
    monkeypatch.setattr(meta_classifier_features, "FEATURE_DIM", 2)
    vector = _base_feature_vector({"indicators": {"hurst": 0.1, "adx": 0.2, "vol_ratio": 0.3}})
    assert vector == pytest.approx([0.1, 0.2])


def test_extract_meta_feature_vector_truncates_when_vector_exceeds_meta_dim(monkeypatch):
    monkeypatch.setattr(meta_classifier_features, "FEATURE_DIM", 3)
    monkeypatch.setattr(meta_classifier_features, "META_FEATURE_DIM", 5)
    metrics = {
        "feature_vector": [1.0, 2.0, 3.0],
        "cross_symbol_features": {
            "cross_symbol_prob_delta": 0.4,
            "cross_symbol_vol_ratio_diff": 0.2,
            "cross_symbol_rsi_spread": 0.1,
        },
    }
    vector = extract_meta_feature_vector(metrics)
    assert len(vector) == 5
    assert vector == pytest.approx([1.0, 2.0, 3.0, 0.4, 0.2])


def test_extract_meta_feature_vector_pads_when_vector_shorter_than_meta_dim(monkeypatch):
    monkeypatch.setattr(meta_classifier_features, "FEATURE_DIM", 2)
    monkeypatch.setattr(meta_classifier_features, "META_FEATURE_DIM", 8)
    vector = extract_meta_feature_vector({"feature_vector": [0.7, 0.3]})
    assert len(vector) == 8
    assert vector[:2] == pytest.approx([0.7, 0.3])
    assert all(v == 0.0 for v in vector[2:])


def test_finalize_meta_vector_pads_short_payload():
    padded = meta_classifier_features._finalize_meta_vector([1.0, 2.0])
    assert len(padded) == META_FEATURE_DIM
    assert padded[0] == pytest.approx(1.0)
    assert padded[1] == pytest.approx(2.0)
    assert padded[-1] == 0.0


def test_finalize_meta_vector_truncates_long_payload():
    long_vector = [float(i) for i in range(META_FEATURE_DIM + 5)]
    truncated = meta_classifier_features._finalize_meta_vector(long_vector)
    assert len(truncated) == META_FEATURE_DIM
    assert truncated[15] == 3.0


def test_base_feature_vector_pads_when_feature_dim_large(monkeypatch):
    monkeypatch.setattr(meta_classifier_features, "FEATURE_DIM", 20)
    vector = meta_classifier_features._base_feature_vector({"indicators": {}})
    assert len(vector) == 20
    assert vector[-1] == 0.0
