from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from tests.market_symbols import ALL_SYMBOLS


def _build_orch(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        return Orchestrator(orch_config, "token")


@pytest.mark.asyncio
async def test_execute_cluster_suspends_trading_during_training_phase(orch_config_train):
    orch = _build_orch(orch_config_train)
    orch._active_cycle_id = 2
    orch._dl_training_symbols = frozenset(ALL_SYMBOLS[:2])
    with (
        patch.object(orch.executor, "_collect_orders") as mock_collect,
        patch.object(orch.executor.logger, "info") as mock_info,
    ):
        await orch.executor.execute_cluster({})
    mock_collect.assert_not_called()
    assert any("FASE TREINO" in str(c) for c in mock_info.call_args_list)
    assert orch._dl_training_phase is True


@pytest.mark.asyncio
async def test_execute_cluster_training_phase_dedupes_repeated_log(orch_config_train):
    orch = _build_orch(orch_config_train)
    orch._active_cycle_id = 3
    orch._dl_training_symbols = frozenset({"X_1"})
    with patch.object(orch.executor.logger, "info") as mock_info:
        await orch.executor.execute_cluster({})
        await orch.executor.execute_cluster({})
    phase_logs = [c for c in mock_info.call_args_list if "FASE TREINO" in str(c)]
    assert len(phase_logs) == 1


@pytest.mark.asyncio
async def test_execute_cluster_logs_operation_phase_once_after_training(orch_config):
    orch = _build_orch(orch_config)
    orch._active_cycle_id = 4
    orch._dl_training_phase = True
    orch._dl_training_symbols = frozenset()
    with (
        patch.object(orch.executor, "_collect_orders", return_value=[]),
        patch.object(orch.executor.logger, "info") as mock_info,
    ):
        await orch.executor.execute_cluster({})
        await orch.executor.execute_cluster({})
    phase_logs = [c for c in mock_info.call_args_list if "FASE OPERACAO" in str(c)]
    assert len(phase_logs) == 1
    assert orch._dl_training_phase is False


@pytest.mark.asyncio
async def test_execute_cluster_logs_training_phase_when_models_pending(orch_config):
    orch = _build_orch(orch_config)
    orch._active_cycle_id = 6
    orch._dl_training_symbols = frozenset(ALL_SYMBOLS[:1])
    with (
        patch.object(orch.executor, "_collect_orders") as mock_collect,
        patch.object(orch.executor.logger, "info") as mock_info,
    ):
        await orch.executor.execute_cluster({})
    mock_collect.assert_not_called()
    assert any("FASE TREINO" in str(c) for c in mock_info.call_args_list)
    assert not any("MODELO AUSENTE" in str(c) for c in mock_info.call_args_list)


@pytest.mark.asyncio
async def test_execute_cluster_without_phase_attributes_runs_normally(orch_config):
    orch = _build_orch(orch_config)
    orch._active_cycle_id = 5
    with (
        patch.object(orch.executor, "_collect_orders", return_value=[]) as mock_collect,
        patch.object(orch.executor.logger, "info") as mock_info,
    ):
        await orch.executor.execute_cluster({})
    mock_collect.assert_called_once()
    assert not any("FASE" in str(c) for c in mock_info.call_args_list)
