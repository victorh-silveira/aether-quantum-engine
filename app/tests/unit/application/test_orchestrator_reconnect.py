import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_orchestrator_run_reconnect_fails_sleeps_backoff(orch_config):
    TradingState.reset()
    sleeps: list[float] = []

    async def track_sleep(delay: float) -> None:
        sleeps.append(delay)
        if delay == 8.0:
            orch.ws.is_running = True
        if len(sleeps) >= 12:
            orch.running = False

    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.is_running = False
        orch._setup_session = AsyncMock(side_effect=[True, False, True])
        orch._start_streams = AsyncMock(return_value=True)
        orch.running = True
        with (
            patch(
                "src.application.services.orchestrator.run_initial_bootstrap_training",
                new_callable=AsyncMock,
            ),
            patch("src.application.services.orchestrator.asyncio.sleep", side_effect=track_sleep),
        ):
            await asyncio.wait_for(orch.run(), timeout=5.0)
    assert 8.0 in sleeps


@pytest.mark.asyncio
async def test_orchestrator_run_reconnect_success_logs_recovery(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.ws.is_running = False
        orch._setup_session = AsyncMock(return_value=True)
        stream_calls = {"n": 0}

        async def restore_ws_and_streams():
            stream_calls["n"] += 1
            if stream_calls["n"] >= 2:
                orch.ws.is_running = True
            return True

        orch._start_streams = AsyncMock(side_effect=restore_ws_and_streams)
        orch.running = True
        loops = {"n": 0}

        async def stop_after_recovery_sleep(_delay: float) -> None:
            loops["n"] += 1
            if loops["n"] >= 3:
                orch.running = False

        with (
            patch(
                "src.application.services.orchestrator.run_initial_bootstrap_training",
                new_callable=AsyncMock,
            ),
            patch(
                "src.application.services.orchestrator.asyncio.sleep",
                side_effect=stop_after_recovery_sleep,
            ),
            patch.object(orch.logger, "info") as mock_info,
        ):
            await asyncio.wait_for(orch.run(), timeout=5.0)
        assert any("RECOV: WebSocket restaurado." in str(c) for c in mock_info.call_args_list)
