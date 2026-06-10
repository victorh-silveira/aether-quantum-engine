from unittest.mock import AsyncMock, MagicMock, patch

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.execution_blockers import blocked_metrics_brief
from src.domain.models.trade import TradeDirection


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
        assert any("EXEC_NONE" in c and "R_75" in c for c in calls)
        assert not any("EXEC_NONE" in c and "R_50" in c for c in calls)


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


def test_blocked_metrics_brief_formats_values_and_handles_empty():
    metrics = {"trade_score": 0.57, "raw_prob": 0.46, "val_accuracy": 0.53, "val_brier": 0.31}
    assert blocked_metrics_brief(metrics) == " s0.57 r0.54 v0.53 b0.31"
    assert blocked_metrics_brief({"gate_reason": "conviction"}) == ""


def test_log_execution_blockers_includes_metric_values(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 5
        orch.symbols = ["R_50"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {
                    "R_50": {
                        "direction": TradeDirection.PUT,
                        "metrics": {
                            "execute": False,
                            "gate_reason": "brier",
                            "trade_score": 0.57,
                            "raw_prob": 0.46,
                            "val_accuracy": 0.53,
                            "val_brier": 0.31,
                        },
                    }
                },
            )
        assert "R_50:brier s0.57 r0.54 v0.53 b0.31" in mock_info.call_args.args[2]


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
