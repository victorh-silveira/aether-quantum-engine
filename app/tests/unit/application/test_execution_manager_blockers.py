from unittest.mock import AsyncMock, MagicMock, patch

from src.application.services.orchestrator import Orchestrator


def test_log_execution_blockers_groups_training_symbols(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 6
        orch.symbols = ["RDBULL", "RDBEAR"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {
                    "RDBULL": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
                    "RDBEAR": {
                        "direction": None,
                        "metrics": {"gate_reason": "direction_margin", "execute": False},
                    },
                },
            )
        calls = [str(c) for c in mock_info.call_args_list]
        assert any("DL_TREINO" in c and "RDBULL" in c for c in calls)
        assert not any("EXEC_NONE" in c for c in calls)


def test_log_execution_blockers_training_dedupe_and_completion(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 7
        orch.symbols = ["RDBULL"]
        training = {"RDBULL": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}}}
        with (
            patch.object(orch.executor.logger, "info") as mock_info,
            patch.object(orch.executor.logger, "debug") as mock_debug,
        ):
            orch.executor._log_execution_blockers(training)
            orch.executor._log_execution_blockers(training)
        assert sum("DL_TREINO" in str(c) for c in mock_info.call_args_list) == 1
        assert any("DL_TREINO" in str(c) for c in mock_debug.call_args_list)
        trained = {"RDBULL": {"direction": None, "metrics": {"gate_reason": "conviction", "execute": False}}}
        with patch.object(orch.executor.logger, "info") as mock_info_done:
            orch.executor._log_execution_blockers(trained)
        assert any("concluido" in str(c) for c in mock_info_done.call_args_list)


def test_log_execution_blockers_skips_symbol_without_decision_entry(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 4
        orch.symbols = ["RDBULL", "RDBEAR"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {"RDBULL": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}}},
            )
        assert mock_info.call_args_list == []
