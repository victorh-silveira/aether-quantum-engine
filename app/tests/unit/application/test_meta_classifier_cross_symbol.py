import pytest

from src.application.services.meta_classifier_cross_symbol import (
    CROSS_SYMBOL_KEYS,
    attach_cross_symbol_features_to_decisions,
    compute_cross_symbol_triplet,
)
from src.application.services.meta_classifier_features import META_FEATURE_DIM, extract_meta_feature_vector
from src.domain.models.trade import TradeDirection


def _metrics(*, prob: float, rsi: float, vol_ratio: float) -> dict:
    return {
        "calibrated_prob": prob,
        "micro_indicators": {"rsi": rsi, "vol_ratio": vol_ratio},
        "feature_vector": [0.1] * 34,
    }


def test_compute_cross_symbol_triplet_values():
    triplet = compute_cross_symbol_triplet(
        _metrics(prob=0.62, rsi=58.0, vol_ratio=1.05),
        _metrics(prob=0.41, rsi=44.0, vol_ratio=0.92),
    )
    assert triplet["cross_symbol_prob_delta"] == pytest.approx(abs(0.62 - (1.0 - 0.41)))
    assert triplet["cross_symbol_vol_ratio_diff"] == pytest.approx(0.13)
    assert triplet["cross_symbol_rsi_spread"] == pytest.approx(14.0)


def test_compute_cross_symbol_triplet_defaults_when_missing():
    triplet = compute_cross_symbol_triplet(None, None)
    assert list(triplet.keys()) == list(CROSS_SYMBOL_KEYS)
    assert all(value == 0.0 for value in triplet.values())


def test_compute_cross_symbol_triplet_defaults_prob_when_absent():
    triplet = compute_cross_symbol_triplet(
        {"micro_indicators": {"rsi": 50.0, "vol_ratio": 1.0}},
        {"micro_indicators": {"rsi": 50.0, "vol_ratio": 1.0}},
    )
    assert triplet["cross_symbol_prob_delta"] == pytest.approx(0.0)


def test_compute_cross_symbol_triplet_falls_back_to_indicators_bucket():
    triplet = compute_cross_symbol_triplet(
        {"micro_indicators": "invalid", "indicators": {"rsi": 70.0, "vol_ratio": 1.2}},
        {"micro_indicators": "invalid", "indicators": {"rsi": 30.0, "vol_ratio": 0.8}},
    )
    assert triplet["cross_symbol_rsi_spread"] == pytest.approx(40.0)
    assert triplet["cross_symbol_vol_ratio_diff"] == pytest.approx(0.4)


def test_attach_cross_symbol_features_to_decisions():
    decisions = {
        "OTC_SPC": {"direction": TradeDirection.CALL, "metrics": _metrics(prob=0.66, rsi=60.0, vol_ratio=1.1)},
    }
    attach_cross_symbol_features_to_decisions(decisions)
    metrics = decisions["OTC_SPC"]["metrics"]
    assert metrics["cross_symbol_features"]["cross_symbol_rsi_spread"] == pytest.approx(0.0)
    assert metrics["cross_symbol_features"]["cross_symbol_prob_delta"] == pytest.approx(0.0)
    vector = extract_meta_feature_vector(metrics)
    assert len(vector) == META_FEATURE_DIM
    assert vector[-3] == pytest.approx(0.0)


def test_extract_meta_feature_vector_uses_meta_feature_vector_cache():
    cached = [float(i) for i in range(META_FEATURE_DIM)]
    metrics = {"meta_feature_vector": cached}
    vector = extract_meta_feature_vector(metrics)
    assert len(vector) == META_FEATURE_DIM
    assert vector[35] == 3.0
    assert vector[37] == 3.0
    assert vector[0] == 0.0
    assert vector[34] == 34.0
    assert metrics["meta_feature_vector"] is vector
