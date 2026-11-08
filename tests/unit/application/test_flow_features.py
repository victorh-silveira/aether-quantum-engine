"""Testes unitarios para flow_features (100% cobertura)."""

import numpy as np
import pytest

from src.application.services.flow_features import (
    FLOW_FEATURE_COUNT,
    FLOW_FEATURE_KEYS,
    compute_keltner_deviation_ratio,
    flow_feature_pair_from_metrics,
    flow_features_from_micro_series,
)


def test_compute_keltner_deviation_ratio() -> None:
    res = compute_keltner_deviation_ratio(105.0, ema=100.0, upper=110.0, lower=90.0)
    assert res == pytest.approx(0.05)

    res_zero_mid = compute_keltner_deviation_ratio(5.0, ema=0.0, upper=10.0, lower=-10.0)
    assert res_zero_mid == pytest.approx(0.25)


def test_flow_features_from_micro_series_short_array() -> None:
    res = flow_features_from_micro_series(np.array([1.0, 2.0]), granularity=900, symbol="OTC_SSMI")
    assert res == dict.fromkeys(FLOW_FEATURE_KEYS, 0.0)

    res_none = flow_features_from_micro_series(None, granularity=900, symbol="OTC_SSMI")
    assert res_none == dict.fromkeys(FLOW_FEATURE_KEYS, 0.0)


def test_flow_features_from_micro_series_valid() -> None:
    closes = np.linspace(100.0, 110.0, 30)
    res = flow_features_from_micro_series(closes, granularity=900, symbol="OTC_SSMI")
    assert "micro_tick_acceleration" in res
    assert "keltner_deviation_ratio" in res
    assert isinstance(res["micro_tick_acceleration"], float)
    assert isinstance(res["keltner_deviation_ratio"], float)


def test_flow_feature_pair_from_metrics() -> None:
    pair = flow_feature_pair_from_metrics(
        {"flow_features": {"micro_tick_acceleration": 0.5, "keltner_deviation_ratio": 0.1}}
    )
    assert pair == [0.5, 0.1]

    pair_empty = flow_feature_pair_from_metrics({})
    assert pair_empty == [0.0] * FLOW_FEATURE_COUNT
