from unittest.mock import AsyncMock, MagicMock, patch

from src.application.services.orchestrator import Orchestrator


def test_log_execution_blockers_data_gate(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 2
        orch.symbols = ["R_50"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {"R_50": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}}},
            )
        assert "dados" in mock_info.call_args.args[2]


def test_log_execution_blockers_direction_margin_no_raw(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 2
        orch.symbols = ["R_50"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {
                    "R_50": {
                        "direction": None,
                        "metrics": {"gate_reason": "direction_margin", "execute": False},
                    }
                },
            )
        assert mock_info.call_args.args[2] == "R_50:sem_direcao"


def test_log_execution_blockers_direction_margin_raw(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 2
        orch.symbols = ["R_50"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {
                    "R_50": {
                        "direction": None,
                        "metrics": {"gate_reason": "direction_margin", "raw_prob": 0.51, "execute": False},
                    }
                },
            )
        assert "sem_direcao:r0.51" in mock_info.call_args.args[2]


def test_log_execution_blockers_skips_symbol_without_decision_entry(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 4
        orch.symbols = ["R_50", "R_75"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {"R_50": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}}},
            )
        assert "dados" in mock_info.call_args.args[2]


def test_log_execution_blockers_sem_direcao_generic_gate(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 3
        orch.symbols = ["R_50"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {"R_50": {"direction": None, "metrics": {"gate_reason": "conviction", "execute": False}}},
            )
        assert "R_50:sem_direcao" in mock_info.call_args.args[2]
