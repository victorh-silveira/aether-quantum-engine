from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from src.application.services.orchestrator.orchestrator_data_signature import (
    get_data_state_signature,
    m1_boundary_epoch,
)
from src.application.services.orchestrator.trading_cycle_entry import (
    _log_market_signature_invalidation,
    trading_cycle_entry_allowed,
)
from src.domain.models.market_data import Candle


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"

_MACRO_EPOCH = 1_700_000_900
_MICRO_EPOCH = 1_700_001_000


def _bear_candle(epoch: int) -> Candle:
    return Candle("RDBEAR", 1.0, 1.1, 0.9, 1.05, datetime.now(), epoch)


def _seed_dual_timeframe_stream(orch) -> None:
    orch.stream.macro_candles = {"RDBEAR": [_bear_candle(_MACRO_EPOCH)]}
    orch.stream.micro_candles = {"RDBEAR": [_bear_candle(_MICRO_EPOCH)]}


def test_get_data_state_signature_empty_without_stream():
    orch = SimpleNamespace(symbols=["RDBEAR"], stream=None)
    assert get_data_state_signature(orch) == ""


def test_dual_timeframe_data_signature(orch_ready):
    orch = orch_ready
    _seed_dual_timeframe_stream(orch)
    sig = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 15))
    assert sig.startswith("m1b:")
    assert ";m1:" in sig
    assert ";m15:" in sig
    assert f"RDBEAR@{_MACRO_EPOCH}" in sig
    assert f"RDBEAR@{_MICRO_EPOCH}" in sig


def test_data_signature_changes_on_m1_boundary_with_static_m15(orch_ready):
    orch = orch_ready
    _seed_dual_timeframe_stream(orch)
    sig_t0 = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 10))
    sig_t1 = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 70))
    assert sig_t0 != sig_t1
    assert sig_t0.split(";")[0] != sig_t1.split(";")[0]
    assert sig_t0.split(";m15:")[1] == sig_t1.split(";m15:")[1]


def test_data_signature_changes_when_micro_epoch_advances_with_static_m15(orch_ready):
    orch = orch_ready
    _seed_dual_timeframe_stream(orch)
    sig_before = get_data_state_signature(orch, now=float(_MICRO_EPOCH))
    orch.stream.micro_candles = {"RDBEAR": [_bear_candle(_MICRO_EPOCH + 60)]}
    sig_after = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 60))
    assert sig_before != sig_after
    assert f"RDBEAR@{_MACRO_EPOCH}" in sig_before
    assert f"RDBEAR@{_MACRO_EPOCH}" in sig_after


def test_m1_boundary_epoch_aligns_to_minute(orch_ready):
    orch = orch_ready
    orch._last_epoch = _MICRO_EPOCH + 15
    assert m1_boundary_epoch(orch, now=float(_MICRO_EPOCH + 45)) == _MICRO_EPOCH
    assert m1_boundary_epoch(orch, now=float(_MICRO_EPOCH + 75)) == _MICRO_EPOCH + 60


def test_get_data_state_signature_uses_wall_clock_when_now_omitted(orch_ready):
    orch = orch_ready
    _seed_dual_timeframe_stream(orch)
    with patch(
        "src.application.services.orchestrator.orchestrator_data_signature.time.time",
        return_value=float(_MICRO_EPOCH + 30),
    ):
        sig = get_data_state_signature(orch)
    assert sig.startswith(f"m1b:{_MICRO_EPOCH};")


def test_m1_boundary_without_anchor_epoch_uses_clock(orch_ready):
    orch = orch_ready
    orch._last_epoch = 0
    assert m1_boundary_epoch(orch, now=1234.0) == 1200


def test_get_data_state_signature_reads_macro_from_legacy_candles_store(orch_ready):
    orch = orch_ready
    orch.stream.macro_candles = None
    orch.stream.candles = {"RDBEAR": [_bear_candle(_MACRO_EPOCH)]}
    orch.stream.micro_candles = {"RDBEAR": [_bear_candle(_MICRO_EPOCH)]}
    sig = get_data_state_signature(orch, now=float(_MICRO_EPOCH))
    assert f"RDBEAR@{_MACRO_EPOCH}" in sig


def test_get_data_state_signature_empty_when_no_candles(orch_ready):
    orch = orch_ready
    orch.stream.macro_candles = {}
    orch.stream.micro_candles = {}
    assert get_data_state_signature(orch, now=float(_MICRO_EPOCH)) == ""


def test_trading_cycle_entry_allowed_when_m1_boundary_shifts_same_anchor_epoch(orch_ready, caplog):
    orch = orch_ready
    _seed_dual_timeframe_stream(orch)
    orch._last_epoch = _MICRO_EPOCH
    orch._last_processed_epoch = _MICRO_EPOCH
    orch.last_data_signature = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 5))
    with (
        patch(f"{TRADING_CYCLE_MODULE}.time.time", return_value=float(_MICRO_EPOCH + 65)),
        caplog.at_level("DEBUG", logger="AETH"),
    ):
        assert trading_cycle_entry_allowed(orch) is True
    invalidation_logs = [record for record in caplog.records if "DATA_SIG" in record.message]
    assert invalidation_logs
    assert "cache invalidado" in invalidation_logs[0].message


def test_log_market_signature_invalidation_skips_unchanged_signature(orch_ready, caplog):
    orch = orch_ready
    with caplog.at_level("DEBUG", logger="AETH"):
        _log_market_signature_invalidation(orch, previous="sig-a", current="sig-a")
    assert not [record for record in caplog.records if "DATA_SIG" in record.message]


def test_log_market_signature_invalidation_deduplicates_repeated_key(orch_ready, caplog):
    orch = orch_ready
    with caplog.at_level("DEBUG", logger="AETH"):
        _log_market_signature_invalidation(orch, previous="sig-a", current="sig-b")
        _log_market_signature_invalidation(orch, previous="sig-a", current="sig-b")
    assert len([record for record in caplog.records if "DATA_SIG" in record.message]) == 1


def test_trading_cycle_entry_blocked_when_signature_and_epoch_unchanged(orch_ready):
    orch = orch_ready
    _seed_dual_timeframe_stream(orch)
    orch._last_epoch = _MICRO_EPOCH
    orch._last_processed_epoch = _MICRO_EPOCH
    orch.last_data_signature = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 5))
    with patch(f"{TRADING_CYCLE_MODULE}.time.time", return_value=float(_MICRO_EPOCH + 10)):
        assert trading_cycle_entry_allowed(orch) is False
