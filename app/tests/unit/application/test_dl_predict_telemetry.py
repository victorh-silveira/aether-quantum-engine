"""Testes unitarios para stamp_micro_frame_telemetry e stamp_macro_frame_telemetry."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.application.services.deep_learning.dl_predict_telemetry import (
    _series_last,
    stamp_macro_frame_telemetry,
    stamp_micro_frame_telemetry,
)


def test_series_last_edge_cases():
    """Valida retorno padrao e ultimo item da serie."""
    assert _series_last({}, "inexistente", 1.5) == 1.5
    assert _series_last({"vazio": []}, "vazio", 2.0) == 2.0
    assert _series_last({"valores": [10.0, 20.0, 30.5]}, "valores") == 30.5


def test_stamp_macro_frame_telemetry_guards():
    """Valida retornos antecipados em condicoes de guarda de stream e closes."""
    metrics: dict = {}
    stamp_macro_frame_telemetry(None, "R_100", metrics, {})
    assert "macro_indicators" not in metrics

    orch_sem_get = SimpleNamespace(stream=SimpleNamespace())
    stamp_macro_frame_telemetry(orch_sem_get, "R_100", metrics, {})
    assert "macro_indicators" not in metrics

    stream_mock = MagicMock()
    stream_mock.get_numpy_series.return_value = None
    orch = SimpleNamespace(stream=stream_mock)
    stamp_macro_frame_telemetry(orch, "R_100", metrics, {})
    assert "macro_indicators" not in metrics

    stream_mock.get_numpy_series.return_value = np.array([1.0, 2.0, 3.0])
    stamp_macro_frame_telemetry(orch, "R_100", metrics, {})
    assert "macro_indicators" not in metrics


def test_stamp_macro_frame_telemetry_success():
    """Valida preenchimento dos indicadores macro com serie valida."""
    closes = np.linspace(100.0, 110.0, 50)
    stream = SimpleNamespace(
        get_numpy_series=MagicMock(return_value=closes),
        macro_granularity=3600,
    )
    orch = SimpleNamespace(stream=stream)
    metrics: dict = {}
    stamp_macro_frame_telemetry(orch, "R_100", metrics, {"granularity": 3600})
    assert "macro_indicators" in metrics
    for key in ("rsi", "vol_ratio", "adx", "hurst"):
        assert key in metrics["macro_indicators"]


def test_stamp_micro_frame_telemetry_guards():
    """Valida guards quando stream ou dados micro sao invalidos."""
    metrics: dict = {}
    stamp_micro_frame_telemetry(None, "R_100", metrics, {})
    assert "micro_indicators" not in metrics

    orch_sem_micro = SimpleNamespace(stream=SimpleNamespace())
    stamp_micro_frame_telemetry(orch_sem_micro, "R_100", metrics, {})
    assert "micro_indicators" not in metrics

    stream_mock = MagicMock()
    stream_mock.get_micro_numpy_series.return_value = None
    orch = SimpleNamespace(stream=stream_mock, _patched_ohlc={})
    stamp_micro_frame_telemetry(orch, "R_100", metrics, {})
    assert "micro_indicators" not in metrics

    stream_mock.get_micro_numpy_series.return_value = np.array([1.0, 2.0])
    stamp_micro_frame_telemetry(orch, "R_100", metrics, {})
    assert "micro_indicators" not in metrics


def test_stamp_micro_frame_telemetry_with_snap_and_precomputed():
    """Valida caminho com snapshot de OHLC remendado e precomputed_series."""
    n = 20
    snap = {
        "close": np.linspace(100.0, 105.0, n),
        "high": np.linspace(101.0, 106.0, n),
        "low": np.linspace(99.0, 104.0, n),
        "open": np.linspace(100.0, 105.0, n),
    }
    orch = SimpleNamespace(
        _patched_ohlc={"R_100": snap},
        stream=SimpleNamespace(
            get_micro_numpy_series=MagicMock(),
            get_numpy_series=MagicMock(return_value=snap["close"]),
        ),
    )
    precomputed = {
        "rsi": [50.0],
        "vol_ratio_short_long": [1.1],
        "bb_width": [0.05],
        "bb_width_raw": [0.04],
        "micro_bid_ask_spread_momentum": [0.01],
        "micro_bid_ask_spread_momentum_zscore": [0.2],
        "volatility_shadow_ratio": [0.8],
        "volatility_shadow_ratio_zscore": [0.5],
        "price_velocity": [0.3],
        "tick_count": [12.0],
    }
    metrics: dict = {}
    stamp_micro_frame_telemetry(orch, "R_100", metrics, {"micro_granularity": 180}, precomputed_series=precomputed)
    assert "micro_indicators" in metrics
    assert "flow_features" in metrics
    assert metrics["micro_indicators"]["rsi"] == 50.0
    assert metrics["flow_features"]["micro_bid_ask_spread_momentum"] == 0.01


def test_stamp_micro_frame_telemetry_fallback_stream_and_tick_buffer():
    """Valida caminho sem snapshot de vela e com tick_buffer fornecendo aceleracao."""
    n = 30
    closes = np.linspace(100.0, 105.0, n)
    highs = closes + 1.0
    lows = closes - 1.0
    opens = closes - 0.2

    def mock_get(sym, key):
        mapping = {"close": closes, "high": highs, "low": lows, "open": opens}
        return mapping.get(key)

    tick_buf = SimpleNamespace(live_tick_acceleration=MagicMock(return_value=1.75))
    stream = SimpleNamespace(
        get_micro_numpy_series=MagicMock(side_effect=mock_get),
        get_numpy_series=MagicMock(return_value=closes),
        tick_buffer=tick_buf,
    )
    orch = SimpleNamespace(stream=stream, _patched_ohlc={})
    metrics: dict = {}
    stamp_micro_frame_telemetry(orch, "R_100", metrics, {"micro_granularity": 180})
    assert "micro_indicators" in metrics
    assert "flow_features" in metrics
    assert metrics["flow_features"]["micro_tick_acceleration"] == 1.75
