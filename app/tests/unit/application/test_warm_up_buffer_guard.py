import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.application.services.orchestrator.warm_up_buffer_guard import (
    _WARM_UP_GUARD_LOG_MESSAGE,
    STREAM_WARM_UP_DELAY_SECONDS,
    log_warm_up_guard_suspension,
    resolve_stream_warm_up_delay_seconds,
    schedule_stream_warm_up_barrier,
    stream_warm_up_active,
    stream_warm_up_remaining,
    trading_cycle_warm_up_suspended,
)


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"


def test_resolve_stream_warm_up_delay_seconds_defaults_to_45():
    assert resolve_stream_warm_up_delay_seconds({}) == pytest.approx(STREAM_WARM_UP_DELAY_SECONDS)


def test_resolve_stream_warm_up_delay_seconds_reads_config():
    config = {"orchestrator": {"stream_warm_up_delay_seconds": 30}}
    assert resolve_stream_warm_up_delay_seconds(config) == pytest.approx(30.0)


def test_resolve_stream_warm_up_delay_seconds_clamps_negative():
    config = {"orchestrator": {"stream_warm_up_delay_seconds": -5}}
    assert resolve_stream_warm_up_delay_seconds(config) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_schedule_stream_warm_up_barrier_sets_deadline():
    orch = SimpleNamespace(config={"orchestrator": {"stream_warm_up_delay_seconds": 45}}, logger=MagicMock())
    loop = asyncio.get_running_loop()
    base = loop.time()
    delay = schedule_stream_warm_up_barrier(orch)
    assert delay == pytest.approx(45.0)
    assert orch._stream_warmed_up_at - base == pytest.approx(45.0, abs=0.05)
    assert stream_warm_up_active(orch, now=base + 10.0) is True
    assert stream_warm_up_active(orch, now=base + 45.01) is False
    assert stream_warm_up_remaining(orch, now=base + 10.0) == pytest.approx(35.0, abs=0.05)


def test_trading_cycle_warm_up_suspended_returns_none_when_inactive():
    orch = SimpleNamespace(_stream_warmed_up_at=0.0, logger=MagicMock())
    assert trading_cycle_warm_up_suspended(orch) is None


def test_stream_warm_up_remaining_zero_when_inactive():
    orch = SimpleNamespace(_stream_warmed_up_at=0.0)
    assert stream_warm_up_remaining(orch) == 0.0


def test_log_warm_up_guard_suspension_skips_when_inactive(orch_ready, caplog):
    orch = orch_ready
    orch._stream_warmed_up_at = 0.0
    with caplog.at_level("INFO", logger="AETH"):
        log_warm_up_guard_suspension(orch)
    assert caplog.records == []


@pytest.mark.asyncio
async def test_trading_cycle_warm_up_suspended_deduplicates_log(orch_ready, caplog):
    orch = orch_ready
    loop = asyncio.get_running_loop()
    orch._stream_warmed_up_at = loop.time() + 45.0
    with caplog.at_level("INFO", logger="AETH"):
        assert trading_cycle_warm_up_suspended(orch) == SIGNAL_SUSPENDED
        assert trading_cycle_warm_up_suspended(orch) == SIGNAL_SUSPENDED
    guard_logs = [record for record in caplog.records if record.message == _WARM_UP_GUARD_LOG_MESSAGE]
    assert len(guard_logs) == 1


@pytest.mark.asyncio
async def test_trading_cycle_skips_inference_during_warm_up_window(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    loop = asyncio.get_running_loop()
    base = loop.time()
    schedule_stream_warm_up_barrier(orch)
    with patch(
        f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
        new_callable=AsyncMock,
    ) as collect_mock:
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is True
    collect_mock.assert_not_awaited()
    assert stream_warm_up_active(orch, now=base + 20.0) is True


@pytest.mark.asyncio
async def test_trading_cycle_resumes_after_warm_up_window(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    loop = asyncio.get_running_loop()
    base = loop.time()
    orch._stream_warmed_up_at = base + 45.0
    orch.executor.execute_cluster = AsyncMock()
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ) as collect_mock,
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch.object(loop, "time", return_value=base + 46.0),
    ):
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is True
    collect_mock.assert_awaited_once()
    orch.executor.execute_cluster.assert_awaited_once()
