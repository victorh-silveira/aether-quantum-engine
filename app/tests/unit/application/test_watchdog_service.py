import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.watchdog_service import (
    AetherWatchdog,
    WatchdogState,
    build_watchdog,
    start_ingestion_watchdog,
    stop_ingestion_watchdog,
    stream_reconnect,
)


@pytest.mark.asyncio
async def test_watchdog_stale_triggers_reconnect():
    orch = MagicMock()
    orch.running = True
    orch.ws.is_running = True
    orch.stream.is_synchronized = True
    orch.stream.tick_buffer.last_tick_monotonic = MagicMock(return_value=0.0)
    orch._stream_ready_mono = 0.0
    orch._save_full_state = AsyncMock()
    orch.stream.reconnect_stream = AsyncMock(return_value=True)

    watchdog = AetherWatchdog(orch, stale_seconds=30.0, poll_interval=0.01)
    loop = asyncio.get_running_loop()
    orch.stream.tick_buffer.last_tick_monotonic = MagicMock(return_value=loop.time() - 35.0)

    await watchdog._evaluate()

    assert watchdog.state == WatchdogState.HEALTHY
    orch._save_full_state.assert_awaited_once()
    orch.stream.reconnect_stream.assert_awaited_once_with(orch)


@pytest.mark.asyncio
async def test_watchdog_healthy_when_tick_recent():
    orch = MagicMock()
    orch.running = True
    orch.ws.is_running = True
    orch.stream.is_synchronized = True
    loop = asyncio.get_running_loop()
    orch.stream.tick_buffer.last_tick_monotonic = MagicMock(return_value=loop.time() - 2.0)
    orch.stream.reconnect_stream = AsyncMock()

    watchdog = AetherWatchdog(orch, stale_seconds=30.0, poll_interval=0.01)
    await watchdog._evaluate()

    assert watchdog.state == WatchdogState.HEALTHY
    orch.stream.reconnect_stream.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_stop_ingestion_watchdog():
    orch = MagicMock()
    orch.config = {
        "orchestrator": {
            "watchdog_enabled": True,
            "watchdog_stale_tick_seconds": 30,
            "watchdog_poll_interval_seconds": 60,
        }
    }
    orch.running = True
    await start_ingestion_watchdog(orch)
    assert isinstance(orch._ingestion_watchdog, AetherWatchdog)
    await stop_ingestion_watchdog(orch)
    assert orch._ingestion_watchdog is None


def test_build_watchdog_disabled():
    orch = MagicMock()
    orch.config = {"orchestrator": {"watchdog_enabled": False}}
    assert build_watchdog(orch) is None


@pytest.mark.asyncio
async def test_watchdog_start_idempotent():
    orch = MagicMock()
    orch.running = False
    watchdog = AetherWatchdog(orch, stale_seconds=30.0, poll_interval=100.0)
    await watchdog.start()
    first = watchdog._task
    await watchdog.start()
    assert watchdog._task is first
    await watchdog.stop()


@pytest.mark.asyncio
async def test_watchdog_stop_noop_without_task():
    watchdog = AetherWatchdog(MagicMock())
    await watchdog.stop()


@pytest.mark.asyncio
async def test_watchdog_run_loop_cancelled():
    orch = MagicMock()
    orch.running = True
    watchdog = AetherWatchdog(orch, poll_interval=0.001)
    iterations = 0

    async def fake_sleep(_sec: float) -> None:
        nonlocal iterations
        iterations += 1
        if iterations >= 2:
            raise asyncio.CancelledError()

    with (
        patch(
            "src.application.services.orchestrator.watchdog_service.asyncio.sleep",
            side_effect=fake_sleep,
        ),
        patch.object(watchdog, "_evaluate", new_callable=AsyncMock),
        pytest.raises(asyncio.CancelledError),
    ):
        await watchdog._run_loop()


@pytest.mark.asyncio
async def test_watchdog_skips_when_not_running():
    orch = MagicMock()
    orch.running = False
    watchdog = AetherWatchdog(orch)
    await watchdog._evaluate()


@pytest.mark.asyncio
async def test_watchdog_skips_when_stream_unsynced():
    orch = MagicMock()
    orch.running = True
    orch.ws.is_running = True
    orch.stream.is_synchronized = False
    watchdog = AetherWatchdog(orch)
    await watchdog._evaluate()


@pytest.mark.asyncio
async def test_watchdog_grace_period_after_stream_ready():
    orch = MagicMock()
    orch.running = True
    orch.ws.is_running = True
    orch.stream.is_synchronized = True
    loop = asyncio.get_running_loop()
    orch.stream.tick_buffer.last_tick_monotonic = MagicMock(return_value=0.0)
    orch._stream_ready_mono = loop.time() - 5.0
    orch.stream.reconnect_stream = AsyncMock()
    watchdog = AetherWatchdog(orch, stale_seconds=30.0)
    await watchdog._evaluate()
    orch.stream.reconnect_stream.assert_not_awaited()


@pytest.mark.asyncio
async def test_watchdog_skips_when_no_ready_mono_and_no_ticks():
    orch = MagicMock()
    orch.running = True
    orch.ws.is_running = True
    orch.stream.is_synchronized = True
    orch.stream.tick_buffer.last_tick_monotonic = MagicMock(return_value=0.0)
    orch._stream_ready_mono = 0.0
    orch.stream.reconnect_stream = AsyncMock()
    watchdog = AetherWatchdog(orch, stale_seconds=30.0)
    await watchdog._evaluate()
    orch.stream.reconnect_stream.assert_not_awaited()


@pytest.mark.asyncio
async def test_watchdog_clears_stale_when_tick_resumes():
    orch = MagicMock()
    orch.running = True
    orch.ws.is_running = True
    orch.stream.is_synchronized = True
    loop = asyncio.get_running_loop()
    orch.stream.tick_buffer.last_tick_monotonic = MagicMock(return_value=loop.time() - 2.0)
    orch.stream.reconnect_stream = AsyncMock()
    watchdog = AetherWatchdog(orch, stale_seconds=30.0)
    watchdog._state = WatchdogState.STALE_DATA
    await watchdog._evaluate()
    assert watchdog.state == WatchdogState.HEALTHY


@pytest.mark.asyncio
async def test_watchdog_skips_recovery_when_lock_held():
    orch = MagicMock()
    orch.running = True
    orch.ws.is_running = True
    orch.stream.is_synchronized = True
    loop = asyncio.get_running_loop()
    orch.stream.tick_buffer.last_tick_monotonic = MagicMock(return_value=loop.time() - 40.0)
    orch._save_full_state = AsyncMock()
    orch.stream.reconnect_stream = AsyncMock(return_value=True)
    watchdog = AetherWatchdog(orch, stale_seconds=30.0)
    async with watchdog._recovering:
        await watchdog._evaluate()
    orch.stream.reconnect_stream.assert_not_awaited()


@pytest.mark.asyncio
async def test_watchdog_recovers_when_save_state_fails():
    orch = MagicMock()
    orch.running = True
    orch.ws.is_running = True
    orch.stream.is_synchronized = True
    loop = asyncio.get_running_loop()
    orch.stream.tick_buffer.last_tick_monotonic = MagicMock(return_value=loop.time() - 40.0)
    orch._stream_ready_mono = 0.0
    orch._save_full_state = AsyncMock(side_effect=RuntimeError("persist fail"))
    orch.stream.reconnect_stream = AsyncMock(return_value=True)
    watchdog = AetherWatchdog(orch, stale_seconds=30.0)
    await watchdog._evaluate()
    orch.stream.reconnect_stream.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_reconnect_helpers():
    assert await stream_reconnect(MagicMock(stream=None)) is False
    orch = MagicMock()
    orch.stream = MagicMock(spec=[])
    assert await stream_reconnect(orch) is False


@pytest.mark.asyncio
async def test_start_ingestion_watchdog_disabled():
    orch = MagicMock()
    orch.config = {"orchestrator": {"watchdog_enabled": False}}
    await start_ingestion_watchdog(orch)
    assert orch._ingestion_watchdog is None
