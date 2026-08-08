from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from src.application.services.orchestrator.orchestrator_data_signature import (
    get_data_state_signature,
    m1_boundary_epoch,
    m5_boundary_epoch,
)
from src.application.services.orchestrator.trading_cycle_entry import trading_cycle_entry_allowed
from src.application.services.orchestrator.trading_cycle_entry_guards import (
    _log_market_signature_invalidation,
    _signature_epoch_blocks_cycle,
    commit_trading_cycle_data_signature,
)
from src.domain.models.market_data import Candle


_MACRO_EPOCH = 1_700_000_900
_MICRO_EPOCH = 1_700_001_000


def _bear_candle(epoch: int) -> Candle:
    return Candle("R_10", 1.0, 1.1, 0.9, 1.05, datetime.now(), epoch)


def _seed_dual_timeframe_stream(orch) -> None:
    orch.stream.macro_candles = {"R_10": [_bear_candle(_MACRO_EPOCH)]}
    orch.stream.micro_candles = {"R_10": [_bear_candle(_MICRO_EPOCH)]}


def test_get_data_state_signature_empty_without_stream():
    orch = SimpleNamespace(symbols=["R_10"], stream=None)
    assert get_data_state_signature(orch) == ""


def test_dual_timeframe_data_signature(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 300
    _seed_dual_timeframe_stream(orch)
    sig = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 15))
    assert sig.startswith("m5b:")
    assert ";m5:" in sig
    assert ";m15:" in sig
    assert f"R_10@{_MACRO_EPOCH}" in sig
    assert f"R_10@{_MICRO_EPOCH}" in sig


def test_data_signature_changes_on_m5_boundary_with_static_m15(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 300
    _seed_dual_timeframe_stream(orch)
    sig_t0 = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 10))
    sig_t1 = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 310))
    assert sig_t0 != sig_t1
    assert sig_t0.split(";")[0] != sig_t1.split(";")[0]
    assert sig_t0.split(";m15:")[1] == sig_t1.split(";m15:")[1]


def test_data_signature_changes_when_micro_epoch_advances_with_static_m15(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 300
    _seed_dual_timeframe_stream(orch)
    sig_before = get_data_state_signature(orch, now=float(_MICRO_EPOCH))
    orch.stream.micro_candles = {"R_10": [_bear_candle(_MICRO_EPOCH + 300)]}
    sig_after = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 300))
    assert sig_before != sig_after
    assert f"R_10@{_MACRO_EPOCH}" in sig_before
    assert f"R_10@{_MACRO_EPOCH}" in sig_after


def test_m5_boundary_epoch_aligns_to_five_minutes(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 300
    orch._last_epoch = _MICRO_EPOCH + 15
    assert m5_boundary_epoch(orch, now=float(_MICRO_EPOCH + 45)) == _MICRO_EPOCH
    assert m5_boundary_epoch(orch, now=float(_MICRO_EPOCH + 75)) == _MICRO_EPOCH
    assert m5_boundary_epoch(orch, now=float(_MICRO_EPOCH + 350)) == _MICRO_EPOCH + 300
    assert m1_boundary_epoch(orch, now=float(_MICRO_EPOCH + 45)) == m5_boundary_epoch(
        orch, now=float(_MICRO_EPOCH + 45)
    )


def test_get_data_state_signature_uses_wall_clock_when_now_omitted(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 300
    _seed_dual_timeframe_stream(orch)
    with patch(
        "src.application.services.orchestrator.orchestrator_data_signature.time.time",
        return_value=float(_MICRO_EPOCH + 30),
    ):
        sig = get_data_state_signature(orch)
    assert sig.startswith(f"m5b:{_MICRO_EPOCH};")


def test_m5_boundary_without_anchor_epoch_uses_clock(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 300
    orch._last_epoch = 0
    assert m5_boundary_epoch(orch, now=1234.0) == 1200


def test_get_data_state_signature_reads_macro_from_legacy_candles_store(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 300
    orch.stream.macro_candles = None
    orch.stream.candles = {"R_10": [_bear_candle(_MACRO_EPOCH)]}
    orch.stream.micro_candles = {"R_10": [_bear_candle(_MICRO_EPOCH)]}
    sig = get_data_state_signature(orch, now=float(_MICRO_EPOCH))
    assert f"R_10@{_MACRO_EPOCH}" in sig


def test_get_data_state_signature_empty_when_no_candles(orch_ready):
    orch = orch_ready
    orch.stream.macro_candles = {}
    orch.stream.micro_candles = {}
    assert get_data_state_signature(orch, now=float(_MICRO_EPOCH)) == ""


def test_trading_cycle_entry_allowed_when_m5_boundary_shifts_same_anchor_epoch(orch_ready, caplog):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    _seed_dual_timeframe_stream(orch)
    orch._last_epoch = _MICRO_EPOCH
    orch._last_processed_epoch = _MICRO_EPOCH
    orch.last_data_signature = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 5))
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_data_signature.time.time",
            return_value=float(_MICRO_EPOCH + 305),
        ),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry_guards.time.time",
            return_value=float(_MICRO_EPOCH + 305),
        ),
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


def test_signature_epoch_blocks_when_epoch_unchanged_without_signature():
    orch = SimpleNamespace(_last_epoch=100, _last_processed_epoch=100)
    assert _signature_epoch_blocks_cycle(orch) is True


def test_trading_cycle_entry_blocked_when_signature_and_epoch_unchanged(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    _seed_dual_timeframe_stream(orch)
    orch._last_epoch = _MICRO_EPOCH
    orch._last_processed_epoch = _MICRO_EPOCH
    orch.last_data_signature = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 5))
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_data_signature.time.time",
            return_value=float(_MICRO_EPOCH + 10),
        ),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry_guards.time.time",
            return_value=float(_MICRO_EPOCH + 10),
        ),
    ):
        assert trading_cycle_entry_allowed(orch) is False


def test_commit_trading_cycle_data_signature_noop_without_signature_api():
    commit_trading_cycle_data_signature(SimpleNamespace())


def test_commit_trading_cycle_data_signature_persists_after_cluster(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 300
    _seed_dual_timeframe_stream(orch)
    orch.last_data_signature = ""
    fixed_sig = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 5))
    orch.get_data_state_signature = lambda: fixed_sig
    commit_trading_cycle_data_signature(orch)
    assert orch.last_data_signature == fixed_sig


def test_signature_epoch_allows_retry_after_skip_without_commit(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 300
    _seed_dual_timeframe_stream(orch)
    orch.config["orchestrator"]["cycle_interval_seconds"] = 0
    orch._last_epoch = _MICRO_EPOCH
    orch._last_processed_epoch = 0
    orch.last_data_signature = ""
    fixed_sig = get_data_state_signature(orch, now=float(_MICRO_EPOCH + 5))
    orch.get_data_state_signature = lambda: fixed_sig
    assert _signature_epoch_blocks_cycle(orch) is False
    assert orch.last_data_signature == ""
    orch.last_data_signature = fixed_sig
    assert _signature_epoch_blocks_cycle(orch) is True
