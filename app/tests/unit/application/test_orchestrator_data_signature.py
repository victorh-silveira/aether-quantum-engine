"""Testes unitarios para o modulo de assinatura de dados e fronteiras temporais."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.application.services.orchestrator.orchestrator_data_signature import (
    at_m5_open_window,
    at_signature_boundary,
    get_data_state_signature,
    m1_boundary_epoch,
    m5_boundary_epoch,
    resolve_signature_boundary_seconds,
    seconds_until_next_signature_boundary,
)


def test_resolve_signature_boundary_seconds_defaults():
    orch_empty = SimpleNamespace(config={})
    assert resolve_signature_boundary_seconds(orch_empty) == 180

    orch_none = None
    assert resolve_signature_boundary_seconds(orch_none) == 180

    orch_explicit = SimpleNamespace(config={"orchestrator": {"signature_boundary_seconds": 300}})
    assert resolve_signature_boundary_seconds(orch_explicit) == 300

    orch_cadence = SimpleNamespace(config={"orchestrator": {"cycle_interval_seconds": 60}})
    assert resolve_signature_boundary_seconds(orch_cadence) == 60


def test_seconds_until_next_signature_boundary_with_explicit_now():
    orch = SimpleNamespace(config={"orchestrator": {"signature_boundary_seconds": 300}})
    assert seconds_until_next_signature_boundary(orch, now=1700000000.0) == pytest.approx(100.0)
    assert seconds_until_next_signature_boundary(orch, now=1700000100.0) == pytest.approx(0.0)


def test_at_signature_boundary_and_open_window():
    orch = SimpleNamespace(config={"orchestrator": {"signature_boundary_seconds": 300}})
    assert at_signature_boundary(orch, now=1700000100.0, tolerance=2.0) is True
    assert at_signature_boundary(orch, now=1700000101.5, tolerance=2.0) is True
    assert at_signature_boundary(orch, now=1700000099.0, tolerance=2.0) is True
    assert at_signature_boundary(orch, now=1700000150.0, tolerance=2.0) is False

    assert at_m5_open_window(orch, now=1700000100.0, tolerance=15.0) is True
    assert at_m5_open_window(orch, now=1700000114.0, tolerance=15.0) is True
    assert at_m5_open_window(orch, now=1700000120.0, tolerance=15.0) is False


def test_m5_and_m1_boundary_epoch():
    orch = SimpleNamespace(
        config={"orchestrator": {"signature_boundary_seconds": 300}},
        _last_epoch=1700000100,
    )
    assert m5_boundary_epoch(orch, now=1700000050.0) == 1700000100
    assert m1_boundary_epoch(orch, now=1700000050.0) == 1700000100


def test_resolve_now_anchored_to_tick_buffer_and_fallback():
    mock_tick_buffer = MagicMock()
    mock_tick_buffer.latest_tick_epoch.return_value = 1700000050.0
    mock_stream = SimpleNamespace(tick_buffer=mock_tick_buffer)

    orch = SimpleNamespace(
        config={"orchestrator": {"signature_boundary_seconds": 300}},
        stream=mock_stream,
        symbols=["1HZ75V"],
        _last_epoch=0,
    )

    with patch(
        "src.application.services.orchestrator.orchestrator_data_signature.time.time",
        return_value=1700000051.0,
    ):
        remaining = seconds_until_next_signature_boundary(orch)
        assert remaining == pytest.approx(50.0)

    mock_tick_buffer.latest_tick_epoch.return_value = 1699990000.0
    with patch(
        "src.application.services.orchestrator.orchestrator_data_signature.time.time",
        return_value=1700000051.0,
    ):
        remaining = seconds_until_next_signature_boundary(orch)
        assert remaining == pytest.approx(49.0)

    orch_epoch = SimpleNamespace(
        config={"orchestrator": {"signature_boundary_seconds": 300}},
        stream=None,
        symbols=["1HZ75V"],
        _last_epoch=1700000050,
    )
    with patch(
        "src.application.services.orchestrator.orchestrator_data_signature.time.time",
        return_value=1700000052.0,
    ):
        remaining = seconds_until_next_signature_boundary(orch_epoch)
        assert remaining == pytest.approx(50.0)


def test_get_data_state_signature_formats_correctly():
    orch_no_stream = SimpleNamespace(stream=None, symbols=["1HZ75V"])
    assert get_data_state_signature(orch_no_stream) == ""

    candle_m5 = SimpleNamespace(epoch=1700000100)
    candle_d1 = SimpleNamespace(epoch=1699920000)
    mock_stream = SimpleNamespace(
        micro_candles={"1HZ75V": [candle_m5]},
        macro_candles={"1HZ75V": [candle_d1]},
    )
    orch = SimpleNamespace(
        config={"orchestrator": {"signature_boundary_seconds": 300}},
        stream=mock_stream,
        symbols=["1HZ75V"],
        _last_epoch=1700000100,
    )
    sig = get_data_state_signature(orch, now=1700000100.0)
    assert sig.startswith("m5b:1700000100;")
    assert "m5:1HZ75V@1700000100" in sig
    assert "m15:1HZ75V@1699920000" in sig
