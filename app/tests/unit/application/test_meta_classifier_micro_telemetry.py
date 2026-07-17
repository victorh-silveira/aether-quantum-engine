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
    stamp_micro_frame_telemetry(_Orch(), "RDBULL", metrics, {"micro_granularity": 300})
    assert "micro_indicators" in metrics
    assert "rsi" in metrics["micro_indicators"]
    assert "vol_ratio" in metrics["micro_indicators"]
    assert metrics["flow_features"]["micro_tick_acceleration"] == 0.17
    assert "keltner_deviation_ratio" in metrics["flow_features"]


def test_stamp_micro_frame_telemetry_noop_without_stream():
    metrics: dict = {}
    stamp_micro_frame_telemetry(object(), "RDBULL", metrics, {})
    assert "micro_indicators" not in metrics
