import asyncio
from datetime import UTC, datetime

import pytest

from src.application.services.infra_timing_config import resolve_orchestrator_timing_config
from src.application.services.orchestrator.api_maintenance_guard import (
    _API_GUARD_LOG_MESSAGE,
    _parse_clock,
    api_maintenance_delay_seconds,
    handle_broker_maintenance_error,
    log_api_maintenance_hibernation,
    orchestrator_api_maintenance_active,
    orchestrator_api_maintenance_remaining,
    schedule_api_maintenance_hibernation,
)


_API_MAINTENANCE_FALLBACK_SECONDS = float(resolve_orchestrator_timing_config()["api_maintenance_fallback_seconds"])


def test_parse_clock_rejects_invalid_values():
    assert _parse_clock("12:30") is None
    assert _parse_clock("aa:00:00") is None
    assert _parse_clock("25:00:00") is None


def test_api_maintenance_delay_seconds_parses_window():
    now = datetime(2026, 7, 7, 0, 0, 3, tzinfo=UTC)
    delay = api_maintenance_delay_seconds(
        "Trading is not available from 00:00:00 to 00:01:00",
        now=now,
    )
    assert delay == pytest.approx(57.0)


def test_api_maintenance_delay_seconds_fallback_without_window():
    assert api_maintenance_delay_seconds("Market is closed") == pytest.approx(_API_MAINTENANCE_FALLBACK_SECONDS)


def test_api_maintenance_delay_seconds_fallback_on_invalid_window():
    assert api_maintenance_delay_seconds("Trading is not available from bad to clock") == pytest.approx(
        _API_MAINTENANCE_FALLBACK_SECONDS
    )


def test_api_maintenance_delay_seconds_fallback_on_invalid_start_clock():
    assert api_maintenance_delay_seconds(
        "Trading is not available from aa:00:00 to 00:01:00",
    ) == pytest.approx(_API_MAINTENANCE_FALLBACK_SECONDS)


def test_api_maintenance_delay_seconds_fallback_on_out_of_range_clock():
    assert api_maintenance_delay_seconds(
        "Trading is not available from 25:00:00 to 00:01:00",
    ) == pytest.approx(_API_MAINTENANCE_FALLBACK_SECONDS)


def test_api_maintenance_delay_seconds_wraps_midnight_window():
    now = datetime(2026, 7, 7, 23, 59, 30, tzinfo=UTC)
    delay = api_maintenance_delay_seconds(
        "Trading is not available from 23:59:00 to 00:01:00",
        now=now,
    )
    assert delay == pytest.approx(90.0)


def test_api_maintenance_delay_seconds_fallback_when_current_equals_end():
    now = datetime(2026, 7, 7, 0, 1, 0, tzinfo=UTC)
    assert api_maintenance_delay_seconds(
        "Trading is not available from 00:00:00 to 00:01:00",
        now=now,
    ) == pytest.approx(_API_MAINTENANCE_FALLBACK_SECONDS)


def test_schedule_api_maintenance_hibernation_ignores_non_maintenance_errors():
    class _Orch:
        _api_maintenance_until = 0.0

    orch = _Orch()
    assert schedule_api_maintenance_hibernation(orch, RuntimeError("network timeout")) == 0.0
    assert orch._api_maintenance_until == 0.0


def test_orchestrator_api_maintenance_remaining_zero_when_inactive(orch_ready):
    orch = orch_ready
    assert orchestrator_api_maintenance_remaining(orch) == 0.0


def test_log_api_maintenance_hibernation_skips_when_inactive(orch_ready, caplog):
    orch = orch_ready
    with caplog.at_level("INFO", logger="AETH"):
        log_api_maintenance_hibernation(orch)
    assert caplog.records == []


@pytest.mark.asyncio
async def test_schedule_api_maintenance_hibernation_sets_deadline(orch_ready):
    orch = orch_ready
    loop = asyncio.get_running_loop()
    base = loop.time()
    delay = schedule_api_maintenance_hibernation(
        orch,
        "Trading is not available from 00:00:00 to 00:01:00",
    )
    assert delay == pytest.approx(_API_MAINTENANCE_FALLBACK_SECONDS)
    assert orch._api_maintenance_until - base == pytest.approx(delay, abs=0.05)
    assert orchestrator_api_maintenance_active(orch, now=base + 1.0) is True
    assert orchestrator_api_maintenance_active(orch, now=base + delay + 0.01) is False
    assert orchestrator_api_maintenance_remaining(orch, now=base + 1.0) == pytest.approx(delay - 1.0, rel=1e-3)


def test_handle_broker_maintenance_error_returns_false_for_other_errors(orch_ready):
    orch = orch_ready
    assert handle_broker_maintenance_error(orch, RuntimeError("network timeout")) is False


def test_log_api_maintenance_hibernation_deduplicates(orch_ready, caplog):
    orch = orch_ready
    orch._api_maintenance_until = 999999.0
    with caplog.at_level("INFO", logger="AETH"):
        log_api_maintenance_hibernation(orch)
        log_api_maintenance_hibernation(orch)
    guard_logs = [record for record in caplog.records if record.message == _API_GUARD_LOG_MESSAGE]
    assert len(guard_logs) == 1
