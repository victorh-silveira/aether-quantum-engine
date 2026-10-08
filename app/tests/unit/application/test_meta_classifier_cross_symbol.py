import pytest

from src.application.services.meta_classifier_cross_symbol import (
    CROSS_SYMBOL_KEYS,
    attach_cross_symbol_features_to_decisions,
    compute_cross_symbol_triplet,
)
from src.application.services.meta_classifier_features import META_FEATURE_DIM, extract_meta_feature_vector
from src.domain.models.trade import TradeDirection


def _metrics(*, vel: float, ticks: float, implied: float) -> dict:
    return {
        "flow_features": {"price_velocity": vel, "tick_count": ticks},
        "indicators": {"implied_vol_ratio": implied},
        "feature_vector": [0.1] * 14,
    }


def test_compute_cross_symbol_triplet_values():
    triplet = compute_cross_symbol_triplet(_metrics(vel=0.4, ticks=150.0, implied=1.2))
    assert triplet["micro_price_velocity"] == pytest.approx(0.4)
    assert triplet["micro_tick_count_norm"] == pytest.approx(0.5)
    assert triplet["implied_vol_centered"] == pytest.approx(0.2)


def test_compute_cross_symbol_triplet_defaults_when_missing():
    triplet = compute_cross_symbol_triplet(None, None)
    assert list(triplet.keys()) == list(CROSS_SYMBOL_KEYS)
    assert all(value == 0.0 for value in triplet.values())


def test_compute_cross_symbol_triplet_defaults_prob_when_absent():
    triplet = compute_cross_symbol_triplet({"micro_indicators": {"rsi": 50.0, "vol_ratio": 1.0}})
    assert triplet["micro_price_velocity"] == pytest.approx(0.0)
    assert triplet["micro_tick_count_norm"] == pytest.approx(0.0)
    assert triplet["implied_vol_centered"] == pytest.approx(0.0)


def test_compute_cross_symbol_triplet_falls_back_to_indicators_bucket():
    triplet = compute_cross_symbol_triplet(
        {"micro_indicators": "invalid", "indicators": {"implied_vol_ratio": 1.4}},
    )
    assert triplet["implied_vol_centered"] == pytest.approx(0.4)


def test_compute_cross_symbol_triplet_coerces_invalid_flow_and_indicators():
    triplet = compute_cross_symbol_triplet(
        {
            "flow_features": {"price_velocity": object(), "tick_count": object()},
            "indicators": {"implied_vol_ratio": object()},
            "micro_indicators": {"price_velocity": object(), "tick_count": object()},
        },
    )
    assert triplet["micro_price_velocity"] == pytest.approx(0.0)
    assert triplet["micro_tick_count_norm"] == pytest.approx(0.0)
    assert triplet["implied_vol_centered"] == pytest.approx(0.0)


def test_attach_cross_symbol_features_to_decisions():
    decisions = {
        "1HZ75V": {
            "direction": TradeDirection.CALL,
            "metrics": _metrics(vel=0.2, ticks=90.0, implied=1.0),
        },
    }
    attach_cross_symbol_features_to_decisions(decisions)
    metrics = decisions["1HZ75V"]["metrics"]
    assert metrics["cross_symbol_features"]["micro_tick_count_norm"] == pytest.approx(0.3)
    assert metrics["cross_symbol_features"]["micro_price_velocity"] == pytest.approx(0.2)
    vector = extract_meta_feature_vector(metrics)
    assert len(vector) == META_FEATURE_DIM
    assert vector[-5] == pytest.approx(0.2)


def test_extract_meta_feature_vector_uses_meta_feature_vector_cache():
    cached = [float(i) for i in range(META_FEATURE_DIM)]
    metrics = {"meta_feature_vector": cached}
    vector = extract_meta_feature_vector(metrics)
    assert len(vector) == META_FEATURE_DIM
    assert vector[15] == 3.0
    assert vector[17] == 3.0
    assert vector[0] == 0.0
    assert vector[14] == 14.0
    assert metrics["meta_feature_vector"] is vector
