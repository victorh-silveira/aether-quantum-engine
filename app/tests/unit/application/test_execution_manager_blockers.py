from unittest.mock import AsyncMock, MagicMock, patch

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import TradeDirection


def test_log_execution_blockers_emits_exec_hold_for_directional_gate(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 8
        orch.symbols = ["R_10"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {
                    "R_10": {
                        "direction": TradeDirection.PUT,
                        "metrics": {
                            "execute": False,
                            "gate_reason": "confidence",
                            "raw_prob": 0.49,
                            "trade_score": 0.51,
                        },
                    },
                },
            )
        assert any("EXEC_HOLD" in str(c) and "R_10:PUT:confidence" in str(c) for c in mock_info.call_args_list)


def test_log_execution_blockers_emits_exec_hold_without_raw_prob(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 10
        orch.symbols = ["R_10"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {
                    "R_10": {
                        "direction": TradeDirection.PUT,
                        "metrics": {"execute": False, "gate_reason": "edge"},
                    },
                },
            )
        assert any("R_10:PUT:edge" in str(c) and "r0." not in str(c) for c in mock_info.call_args_list)


def test_log_execution_blockers_groups_training_symbols(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 6
        orch.symbols = ["R_50", "R_75"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {
                    "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
                    "R_75": {
                        "direction": None,
                        "metrics": {"gate_reason": "direction_margin", "execute": False},
                    },
                },
            )
        calls = [str(c) for c in mock_info.call_args_list]
        assert any("DL_TREINO" in c and "R_50" in c for c in calls)
        assert not any("EXEC_NONE" in c for c in calls)


def test_log_execution_blockers_training_dedupe_and_completion(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 7
        orch.symbols = ["R_50"]
        training = {"R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}}}
        with (
            patch.object(orch.executor.logger, "info") as mock_info,
            patch.object(orch.executor.logger, "debug") as mock_debug,
        ):
            orch.executor._log_execution_blockers(training)
            orch.executor._log_execution_blockers(training)
        assert sum("DL_TREINO" in str(c) for c in mock_info.call_args_list) == 1
        assert any("DL_TREINO" in str(c) for c in mock_debug.call_args_list)
        trained = {"R_50": {"direction": None, "metrics": {"gate_reason": "conviction", "execute": False}}}
        with patch.object(orch.executor.logger, "info") as mock_info_done:
            orch.executor._log_execution_blockers(trained)
        assert any("concluido" in str(c) for c in mock_info_done.call_args_list)


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
        assert mock_info.call_args_list == []


def test_log_execution_blockers_clears_exec_hold_channel(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 9
        orch.symbols = ["R_10"]
        hold = {
            "R_10": {
                "direction": TradeDirection.PUT,
                "metrics": {"execute": False, "gate_reason": "confidence", "raw_prob": 0.49},
            },
        }
        cleared = {
            "R_10": {
                "direction": TradeDirection.CALL,
                "metrics": {"execute": True, "gate_reason": None, "raw_prob": 0.72},
            },
        }
        with (
            patch.object(orch.executor.logger, "info"),
            patch.object(orch.executor.logger, "debug") as mock_debug,
        ):
            orch.executor._log_execution_blockers(hold)
            orch.executor._log_execution_blockers(cleared)
        assert any("EXEC_HOLD || liberado" in str(c) for c in mock_debug.call_args_list)
