from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import TradeDirection
from tests.unit.application.universal_regime_metrics import asymmetric_gate_safe_metrics, bear_put_metrics


@pytest.mark.asyncio
async def test_execution_manager_log_line_contains_exec_and_direction(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        with patch.object(orch.executor.logger, "debug") as mock_dbg:
            orch.executor._log_exec(
                "R_10",
                TradeDirection.CALL,
                1.0,
                {"conviction": 1.0},
                order_n=1,
                contract_id=12345,
            )
        args = mock_dbg.call_args_list[0].args
        assert "ORDEM ENVIADA" in str(args[0])
        assert args[5] == "12345"


def test_execution_manager_mandatory_defaults_when_execution_cfg_not_dict(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.config.setdefault("orchestrator", {})["execution"] = "invalid"
        assert orch.executor._mandatory_trade_each_cycle() is True


def test_execution_manager_collect_orders_mandatory_includes_execute_false(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
        orch.symbols = ["R_10", "R_50"]
        decisions = {
            "R_10": {"direction": TradeDirection.CALL, "metrics": {"conviction": 0.8, "execute": False}},
            "R_50": {
                "direction": TradeDirection.PUT,
                "metrics": bear_put_metrics(conviction=0.9, execute=True),
            },
        }
        orders = orch.executor._collect_orders(decisions)
        assert len(orders) == 1
        assert orders[0][0] == "R_50"


def test_collect_orders_mandatory_bypasses_selection_filter(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.config["deep_learning"] = {
            "selection": {
                "min_conviction_execute": 0.99,
                "min_edge_margin": 0.99,
                "min_val_accuracy": 0.99,
                "strong_raw": 0.99,
                "strong_edge": 0.99,
            }
        }
        orch.symbols = ["R_10"]
        orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["include_anchor_trades"] = True
        orch.config["orchestrator"]["execution"]["mandatory_trade_each_cycle"] = True
        decisions = {
            "R_10": {
                "direction": TradeDirection.CALL,
                "metrics": asymmetric_gate_safe_metrics(
                    execute=True,
                    conviction=0.55,
                    trade_score=0.75,
                    val_accuracy=0.4,
                    edge=0.05,
                    raw_prob=0.56,
                ),
            },
        }
        orders = orch.executor._collect_orders(decisions)
        assert len(orders) == 1


@pytest.mark.asyncio
async def test_execute_cluster_logs_exec_pause_on_stop_win(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 2
        orch.symbols = ["R_10", "R_50"]
        orch.risk_manager.stake_block_reason = MagicMock(return_value="stop_win")
        decisions = {
            "R_10": {
                "direction": TradeDirection.PUT,
                "metrics": asymmetric_gate_safe_metrics(conviction=0.7, execute=True),
            },
        }
        with (
            patch.object(orch.executor.logger, "info") as mock_info,
            patch.object(orch.executor, "_execute_orders", new_callable=AsyncMock, return_value=0) as mock_exec,
        ):
            await orch.executor.execute_cluster(decisions)
        mock_exec.assert_awaited_once()
        assert mock_exec.await_args.args[0] == []
        assert any("EXEC_PAUSE" in str(c) and "stop_win" in str(c) for c in mock_info.call_args_list)


@pytest.mark.asyncio
async def test_execute_orders_skips_zero_stake_without_logging(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 9
        orch.risk_manager.calculate_stake = MagicMock(return_value=0.0)
        orders = [("R_10", TradeDirection.CALL, {"conviction": 0.7})]
        count = await orch.executor._execute_orders(orders, 0.0, 49.0)
        assert count == 0


def test_cluster_stake_block_empty_orders(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        assert orch.executor._cluster_stake_block([], 50.0) is None


def test_log_execution_blockers_silent_on_stake_zero(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 4
        orch.symbols = ["R_10", "R_50"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {"R_10": {"direction": TradeDirection.PUT, "metrics": {"conviction": 0.7, "execute": True}}},
            )
        assert mock_info.call_args_list == []


def test_log_execution_blockers_silent_without_direction(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 3
        orch.symbols = ["R_10", "R_50"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {"R_10": {"direction": None, "metrics": {"execute": True}}},
            )
        assert mock_info.call_args_list == []


@pytest.mark.asyncio
async def test_execute_cluster_runs_with_dl_decisions(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 4
        orch.symbols = ["R_10", "R_50"]
        decisions = {
            "R_10": {
                "direction": TradeDirection.PUT,
                "metrics": asymmetric_gate_safe_metrics(conviction=0.7, execute=True),
            },
        }
        with patch.object(orch.executor, "_execute_orders", new_callable=AsyncMock, return_value=0) as mock_exec:
            await orch.executor.execute_cluster(decisions)
        mock_exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_cluster_executes_when_stake_available(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 5
        orch.symbols = ["R_10", "R_50"]
        orch.risk_manager.calculate_stake = MagicMock(return_value=2.0)
        decisions = {
            "R_10": {
                "direction": TradeDirection.PUT,
                "metrics": bear_put_metrics(conviction=0.7, execute=True),
            },
        }
        with patch.object(orch.executor, "_execute_orders", new_callable=AsyncMock, return_value=1) as mock_exec:
            await orch.executor.execute_cluster(decisions)
        mock_exec.assert_awaited_once()
