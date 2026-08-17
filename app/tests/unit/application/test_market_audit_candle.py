from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.market_audit_candle import (
    candle_binary_side,
    closed_micro_candle_dir_from_stream,
    format_candle_outcome_line,
    last_closed_micro_candle,
    log_closed_candle_outcomes,
    resolve_micro_granularity_seconds,
)
from src.domain.models.market_data import Candle


def _candle(symbol: str, epoch: int, open_: float, close: float) -> Candle:
    return Candle(
        symbol=symbol,
        open=open_,
        high=max(open_, close),
        low=min(open_, close),
        close=close,
        time=datetime.fromtimestamp(epoch, tz=UTC),
        epoch=epoch,
    )


def test_candle_binary_side_call_put_doji():
    assert candle_binary_side(_candle("R_10", 100, 1.0, 1.1)) == "CALL"
    assert candle_binary_side(_candle("R_10", 100, 1.1, 1.0)) == "PUT"
    assert candle_binary_side(_candle("R_10", 100, 1.0, 1.0)) == "DOJI"


def test_closed_micro_candle_dir_from_stream_call_put_doji():
    forming = _candle("R_10", 240, 1.0, 1.0)
    stream_call = SimpleNamespace(micro_candles={"R_10": [_candle("R_10", 120, 1.0, 1.2), forming]})
    stream_put = SimpleNamespace(micro_candles={"R_10": [_candle("R_10", 120, 1.2, 1.0), forming]})
    stream_doji = SimpleNamespace(micro_candles={"R_10": [_candle("R_10", 120, 1.0, 1.0), forming]})
    assert closed_micro_candle_dir_from_stream(stream_call, "R_10") == "CALL"
    assert closed_micro_candle_dir_from_stream(stream_put, "R_10") == "PUT"
    assert closed_micro_candle_dir_from_stream(stream_doji, "R_10") is None
    assert closed_micro_candle_dir_from_stream(None, "R_10") is None


def test_last_closed_micro_candle_uses_penultimate():
    forming = _candle("R_10", 240, 1.0, 1.0)
    closed = _candle("R_10", 120, 1.0, 1.2)
    stream = SimpleNamespace(micro_candles={"R_10": [closed, forming]})
    got = last_closed_micro_candle(stream, "R_10")
    assert got is closed
    assert last_closed_micro_candle(None, "R_10") is None
    assert last_closed_micro_candle(SimpleNamespace(micro_candles={"R_10": [forming]}), "R_10") is None


def test_format_candle_outcome_line_contains_side_and_window():
    candle = _candle("R_10", 1_700_000_000, 10.0, 10.5)
    line = format_candle_outcome_line("R_10", candle, granularity=120)
    assert line.startswith("[CANDLE] || M2 || R_10: CALL |")
    assert "o=10.00000" in line and "c=10.50000" in line
    assert "epoch=1700000000" in line
    assert "->" in line


def test_resolve_micro_granularity_from_stream_and_settings():
    orch = SimpleNamespace(stream=SimpleNamespace(micro_granularity=180), config={})
    assert resolve_micro_granularity_seconds(orch) == 180
    orch2 = SimpleNamespace(stream=None, config={"data_handler": {"micro_granularity": 180}})
    assert resolve_micro_granularity_seconds(orch2) == 180
    assert resolve_micro_granularity_seconds(None) == 60


def test_last_closed_micro_candle_rejects_bad_store():
    assert last_closed_micro_candle(SimpleNamespace(micro_candles="bad"), "R_10") is None
    assert last_closed_micro_candle(SimpleNamespace(micro_candles={"R_10": [1, 2]}), "R_10") is None


def test_resolve_micro_granularity_invalid_falls_back():
    orch = SimpleNamespace(stream=SimpleNamespace(micro_granularity="bad"), config={})
    assert resolve_micro_granularity_seconds(orch) == 60
    orch2 = SimpleNamespace(
        stream=None,
        config={"data_handler": {"micro_granularity": "x"}},
    )
    assert resolve_micro_granularity_seconds(orch2) == 60


def test_format_candle_outcome_line_non_minute_granularity():
    candle = _candle("R_10", 1_700_000_000, 10.0, 9.5)
    line = format_candle_outcome_line("R_10", candle, granularity=90)
    assert "[CANDLE] || 90s || R_10: PUT |" in line


def test_log_closed_candle_outcomes_emits_and_fallbacks():
    closed = _candle("R_10", 120, 1.0, 0.9)
    forming = _candle("R_10", 240, 0.9, 0.9)
    orch = SimpleNamespace(
        stream=SimpleNamespace(micro_candles={"R_10": [closed, forming]}, micro_granularity=120),
        symbols=["R_10"],
        config={"data_handler": {"micro_granularity": 120}},
    )
    logger = MagicMock()
    log_closed_candle_outcomes(logger, None, {"R_10": {}})
    assert logger.info.call_count == 0
    log_closed_candle_outcomes(logger, orch, {"R_10": {}})
    assert logger.info.call_count == 1
    assert "[CANDLE]" in str(logger.info.call_args.args[1])
    assert "PUT" in str(logger.info.call_args.args[1])
    log_closed_candle_outcomes(logger, orch, {"R_10": {}})
    assert logger.info.call_count == 1
    log_closed_candle_outcomes(logger, orch, {})
    assert logger.info.call_count == 1
    empty = SimpleNamespace(stream=SimpleNamespace(micro_candles={"R_10": []}), symbols=["R_10"], config={})
    log_closed_candle_outcomes(MagicMock(), empty, {"R_10": {}})
