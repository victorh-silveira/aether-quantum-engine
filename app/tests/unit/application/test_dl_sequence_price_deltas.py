"""Testes unitarios para calculo de deltas de preco e metricas de movimento."""

from types import SimpleNamespace

import numpy as np

from src.application.services.deep_learning.dl_predict_metrics import (
    attach_dynamic_metrics,
    attach_movement_prediction_metrics,
)
from src.application.services.deep_learning.dl_sequence_extract import sequence_price_deltas


def test_sequence_price_deltas_with_open_array():
    prices = np.array([100.0, 102.0, 105.0, 104.0, 108.0, 110.0], dtype=np.float64)
    open_ = np.array([99.0, 101.0, 103.0, 103.5, 106.0, 109.0], dtype=np.float64)
    lookback = 2
    deltas = sequence_price_deltas(
        prices,
        lookback=lookback,
        label_horizon_bars=1,
        label_smooth_bars=1,
        label_mode="spot_forward",
        open_=open_,
    )
    assert len(deltas) > 0
    expected_first = (104.0 - 103.5) / 103.5
    assert abs(deltas[0] - expected_first) < 1e-5


def test_sequence_price_deltas_without_open():
    prices = np.array([100.0, 102.0, 105.0, 104.0, 108.0, 110.0], dtype=np.float64)
    lookback = 2
    deltas = sequence_price_deltas(
        prices,
        lookback=lookback,
        label_horizon_bars=1,
        label_smooth_bars=1,
        label_mode="spot_forward",
        open_=None,
    )
    assert len(deltas) > 0
    expected_first = (104.0 - 105.0) / 105.0
    assert abs(deltas[0] - expected_first) < 1e-5


def test_attach_movement_prediction_metrics_call_and_put():
    metrics_call = {}
    attach_movement_prediction_metrics(
        metrics_call,
        predicted_delta=0.0050,
        current_direction="CALL",
        atr_norm=0.0025,
    )
    assert metrics_call["predicted_movement_delta"] == 0.0050
    assert abs(metrics_call["expected_drift_pct"] - 0.50) < 1e-5
    assert metrics_call["predicted_movement_side"] == "CALL"
    assert metrics_call["movement_confluence"] is True
    assert abs(metrics_call["movement_atr_ratio"] - 2.0) < 1e-5

    metrics_put = {}
    attach_movement_prediction_metrics(
        metrics_put,
        predicted_delta=-0.0030,
        current_direction="CALL",
        atr_norm=0.0015,
    )
    assert metrics_put["predicted_movement_side"] == "PUT"
    assert metrics_put["movement_confluence"] is False
    assert abs(metrics_put["movement_atr_ratio"] - 2.0) < 1e-5


def test_attach_dynamic_metrics_integrates_movement():
    dummy_model = SimpleNamespace(_last_predicted_delta=0.0018)
    runtime = {
        "model": dummy_model,
        "calibrated_entropy": 0.45,
    }
    metrics = {
        "exec_direction": "CALL",
        "atr_norm": 0.0009,
    }
    attach_dynamic_metrics(
        metrics,
        dynamic=None,
        bb_width=0.01,
        vol_ratio=1.0,
        implied_vol_ratio=1.0,
        symbol="1HZ75V",
        bb_history=[0.01],
        scale_enabled=False,
        runtime=runtime,
    )
    assert metrics.get("predicted_movement_delta") == 0.0018
    assert metrics.get("predicted_movement_side") == "CALL"
    assert metrics.get("movement_confluence") is True
    assert abs(metrics.get("movement_atr_ratio", 0.0) - 2.0) < 1e-5


def test_sequence_price_deltas_breaks_when_future_index_out_of_bounds(monkeypatch):
    prices = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    monkeypatch.setattr(
        "src.application.services.deep_learning.dl_sequence_extract.sequence_labels",
        lambda *args, **kwargs: (np.array([1.0, 1.0]), np.array([1.0, 1.0])),
    )
    deltas = sequence_price_deltas(prices, lookback=1, label_horizon_bars=5)
    assert len(deltas) == 2
