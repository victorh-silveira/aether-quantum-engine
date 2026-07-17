from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_orchestrator_run_early_return_when_setup_fails(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        with patch(
            "src.application.services.orchestrator.orchestrator_run_loop.setup_session",
            AsyncMock(return_value=False),
        ):
            await orch.run()
        assert orch.running is False


@pytest.mark.asyncio
async def test_orchestrator_run_early_return_when_streams_fail(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        with (
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.setup_session",
                AsyncMock(return_value=True),
            ),
            patch(
                "src.application.services.orchestrator.orchestrator_run_loop.start_streams",
                AsyncMock(return_value=False),
            ),
        ):
            await orch.run()
        assert orch.running is False
