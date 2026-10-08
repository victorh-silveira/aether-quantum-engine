import numpy as np

from src.application.services.deep_learning.dl_predict_build import stamp_micro_frame_telemetry
from src.application.services.deep_learning.dl_predict_telemetry import _series_last


class _TickBuffer:
    def live_tick_acceleration(self, _symbol: str) -> float:
        return 0.17


class _Stream:
    def __init__(self):
        self.tick_buffer = _TickBuffer()

    def get_micro_numpy_series(self, _symbol: str, field: str = "close"):
        if field == "close":
            return np.linspace(100.0, 110.0, 32, dtype=np.float64)
        return np.linspace(99.0, 109.0, 32, dtype=np.float64)


class _Orch:
    stream = _Stream()


def test_series_last_defaults_on_missing_or_empty():
    assert _series_last({}, "missing") == 0.0
    assert _series_last({"x": []}, "x", 1.5) == 1.5
    assert _series_last({"x": [2.0, 3.0]}, "x") == 3.0


def test_stamp_micro_frame_telemetry_attaches_micro_indicators():
    metrics: dict = {}
    stamp_micro_frame_telemetry(_Orch(), "R_10", metrics, {"micro_granularity": 300})
    assert "micro_indicators" in metrics
    assert "rsi" in metrics["micro_indicators"]
    assert "vol_ratio" in metrics["micro_indicators"]
    assert metrics["flow_features"]["micro_tick_acceleration"] == 0.17
    assert "keltner_deviation_ratio" in metrics["flow_features"]


def test_stamp_micro_frame_telemetry_noop_without_stream():
    metrics: dict = {}
    stamp_micro_frame_telemetry(object(), "R_10", metrics, {})
    assert "micro_indicators" not in metrics


def test_stamp_micro_frame_telemetry_prefers_patched_snapshot():
    from src.application.services.deep_learning.dl_live_bar_patch import store_patched_ohlc_snapshot

    orch = _Orch()
    closes = np.linspace(200.0, 220.0, 32, dtype=np.float64)
    store_patched_ohlc_snapshot(orch, "R_10", closes, closes - 1.0, closes + 1.0, closes - 2.0)
    metrics: dict = {}
    stamp_micro_frame_telemetry(orch, "R_10", metrics, {"micro_granularity": 120})
    assert "micro_indicators" in metrics
    assert "flow_features" in metrics


def test_stamp_micro_frame_telemetry_uses_precomputed_series():
    from src.application.services.deep_learning.dl_predict_telemetry import stamp_micro_frame_telemetry

    metrics: dict = {}
    series = {
        "rsi": [0.4, 0.55],
        "vol_ratio_short_long": [1.0, 1.2],
        "micro_bid_ask_spread_momentum": [0.1, 0.2],
        "micro_bid_ask_spread_momentum_zscore": [0.0, 0.1],
        "volatility_shadow_ratio": [1.0, 1.1],
        "volatility_shadow_ratio_zscore": [0.0, 0.05],
    }
    stamp_micro_frame_telemetry(
        _Orch(),
        "R_10",
        metrics,
        {"micro_granularity": 300},
        precomputed_series=series,
    )
    assert metrics["micro_indicators"]["rsi"] == 0.55
