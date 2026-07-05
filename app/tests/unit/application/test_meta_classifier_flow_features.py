import numpy as np
import pytest

from src.application.services.meta_classifier_flow_features import (
    FLOW_FEATURE_KEYS,
    compute_keltner_deviation_ratio,
    flow_feature_pair_from_metrics,
    flow_features_from_micro_series,
)


def test_flow_features_from_micro_series_returns_keys():
    closes = np.linspace(100.0, 110.0, 32, dtype=np.float64)
    flow = flow_features_from_micro_series(closes, granularity=60, symbol="RDBULL")
    assert list(flow.keys()) == list(FLOW_FEATURE_KEYS)


def test_flow_features_from_micro_series_short_history_defaults():
    flow = flow_features_from_micro_series(np.array([1.0, 2.0]), granularity=60, symbol="RDBULL")
    assert flow["micro_tick_acceleration"] == pytest.approx(0.0)
    assert flow["keltner_deviation_ratio"] == pytest.approx(0.0)


def test_compute_keltner_deviation_ratio_uses_midline():
    ratio = compute_keltner_deviation_ratio(105.0, ema=100.0, upper=110.0, lower=90.0)
    assert ratio == pytest.approx(0.05)


def test_compute_keltner_deviation_ratio_uses_band_when_mid_zero():
    ratio = compute_keltner_deviation_ratio(102.0, ema=0.0, upper=104.0, lower=100.0)
    assert ratio == pytest.approx(25.5)


def test_flow_features_zero_acceleration_with_single_delta(monkeypatch):
    closes = np.linspace(100.0, 108.0, 8, dtype=np.float64)
    monkeypatch.setattr(
        "src.application.services.meta_classifier_flow_features.precompute_price_series",
        lambda *args, **kwargs: {"keltner_pct_b": np.array([0.55])},
    )
    original_diff = np.diff

    def fake_diff(values):
        if len(values) == 6:
            return np.array([1.0], dtype=np.float64)
        return original_diff(values)

    monkeypatch.setattr(np, "diff", fake_diff)
    flow = flow_features_from_micro_series(closes, granularity=60, symbol="RDBULL")
    assert flow["micro_tick_acceleration"] == pytest.approx(0.0)


def test_flow_feature_pair_from_metrics_reads_stamped_chunk():
    metrics = {
        "flow_features": {
            "micro_tick_acceleration": 0.12,
            "keltner_deviation_ratio": -0.08,
        }
    }
    pair = flow_feature_pair_from_metrics(metrics)
    assert pair == pytest.approx([0.12, -0.08])
