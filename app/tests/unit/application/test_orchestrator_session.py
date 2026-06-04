from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_run_trading_cycle_blocked_by_session(orch_config):
    orch_config["trading"] = {
        "session": {
            "enabled": True,
            "start_hour_utc": 8,
            "end_hour_utc": 18,
            "warmup_seconds_after_stream_sync": 0,
            "minutes_before_close_no_entry": 0,
        }
    }
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch._last_epoch = (1_700_000_000 // 86400) * 86400 + 2 * 3600
        orch.executor.execute_cluster = AsyncMock()
        await orch._run_trading_cycle_if_ready()
        orch.executor.execute_cluster.assert_not_called()


@pytest.mark.asyncio
async def test_run_trading_cycle_ws_not_running_after_lock(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = False
        orch._last_epoch = (1_700_000_000 // 86400) * 86400 + 12 * 3600
        orch.executor.execute_cluster = AsyncMock()
        await orch._run_trading_cycle_if_ready()
        orch.executor.execute_cluster.assert_not_called()


@pytest.mark.asyncio
async def test_run_trading_cycle_collect_failure(orch_config):
    with (
        patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()),
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            side_effect=RuntimeError("dl down"),
        ),
    ):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch._last_epoch = (1_700_000_000 // 86400) * 86400 + 12 * 3600
        orch.executor.execute_cluster = AsyncMock()
        await orch._run_trading_cycle_if_ready()
        orch.executor.execute_cluster.assert_not_called()


@pytest.mark.asyncio
async def test_start_streams_records_ready_at(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.stream.start_candle_stream = AsyncMock()
        assert await orch._start_streams() is True
        assert orch._stream_ready_at is not None
