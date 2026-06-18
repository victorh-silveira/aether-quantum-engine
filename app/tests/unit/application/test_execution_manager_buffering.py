from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.infrastructure.state.trading_state import TradingState


def test_execution_manager_result_buffer_helpers(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.executor._start_result_buffer()
        assert orch._buffer_result_logs is True
        assert orch._pending_result_logs == []
        orch._pending_result_logs = ["a", "b"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._flush_result_buffer()
        assert mock_info.call_count == 2
        assert orch._pending_result_logs == []


@pytest.mark.asyncio
async def test_execution_manager_clears_cuda_cache(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")

        with (
            patch("src.application.services.orchestrator.execution_manager.torch.cuda.is_available", return_value=True),
            patch("src.application.services.orchestrator.execution_manager.torch.cuda.empty_cache") as mock_empty_cache,
        ):
            await orch.executor.execute_cluster({})
            mock_empty_cache.assert_called_once()

        with (
            patch(
                "src.application.services.orchestrator.execution_manager.torch.cuda.is_available", return_value=False
            ),
            patch("src.application.services.orchestrator.execution_manager.torch.cuda.empty_cache") as mock_empty_cache,
        ):
            await orch.executor.execute_cluster({})
            mock_empty_cache.assert_not_called()
