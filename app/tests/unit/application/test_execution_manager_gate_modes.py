from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_execute_cluster_silent_when_not_mandatory_and_execute_false(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 1
        orch.symbols = ["RDBULL"]
        orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
        decisions = {
            "RDBULL": {"direction": TradeDirection.PUT, "metrics": {"conviction": 0.7, "execute": False}},
        }
        with (
            patch.object(orch.executor.logger, "info") as mock_info,
            patch.object(orch.executor, "_execute_orders", new_callable=AsyncMock, return_value=0),
        ):
            await orch.executor.execute_cluster(decisions)
        assert not any("EXEC_NONE" in str(c) for c in mock_info.call_args_list)


@pytest.mark.asyncio
async def test_execute_cluster_mandatory_skips_exec_none_when_execute_false(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 1
        orch.symbols = ["RDBULL"]
        orch.risk_manager.calculate_stake = MagicMock(return_value=2.0)
        orch.risk_manager.stake_block_reason = MagicMock(return_value=None)
        orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
        decisions = {
            "RDBULL": {
                "direction": TradeDirection.CALL,
                "metrics": {
                    "conviction": 0.7,
                    "execute": False,
                    "raw_prob": 0.52,
                    "deploy_ok": True,
                    "val_accuracy": 0.55,
                },
            },
        }
        with (
            patch.object(orch.executor.logger, "info") as mock_info,
            patch.object(orch.executor, "_execute_orders", new_callable=AsyncMock, return_value=1) as mock_exec,
        ):
            await orch.executor.execute_cluster(decisions)
        assert any("EXEC_SEL" in str(c) for c in mock_info.call_args_list)
        mock_exec.assert_awaited_once()
        assert len(mock_exec.await_args.args[0]) == 1


def test_log_execution_blockers_silent_when_gating_blocks(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 2
        orch.symbols = ["RDBULL"]
        orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {
                    "RDBULL": {
                        "direction": TradeDirection.CALL,
                        "metrics": {"execute": False, "gate_reason": "conviction", "conviction": 0.5},
                    },
                }
            )
        assert not any("EXEC_NONE" in str(c) for c in mock_info.call_args_list)


def test_collect_orders_non_mandatory_includes_candidate_with_raw_prob(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
        orch.config.setdefault("deep_learning", {})["min_edge_execute"] = 0.04
        orch.symbols = ["RDBULL"]
        decisions = {
            "RDBULL": {
                "direction": TradeDirection.CALL,
                "metrics": {
                    "execute": True,
                    "deploy_ok": True,
                    "conviction": 0.82,
                    "trade_score": 0.82,
                    "val_accuracy": 0.70,
                    "edge": 0.12,
                    "raw_prob": 0.82,
                    "trend_direction": "CALL",
                    "indicators": {
                        "adx": 0.28,
                        "hurst": 0.55,
                        "vol_ratio": 1.1,
                        "rsi": 0.52,
                        "keltner": 0.55,
                        "cmo": 0.05,
                    },
                },
            },
        }
        orders = orch.executor._collect_orders(decisions)
        assert len(orders) == 1


def test_collect_orders_continuous_keeps_weak_technically_valid_candidate(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.symbols = ["RDBULL"]
        orch.config["deep_learning"] = {
            "min_edge_execute": 0.04,
            "selection": {"min_conviction_execute": 0.99, "min_edge_margin": 0.99, "min_val_accuracy": 0.99},
        }
        decisions = {
            "RDBULL": {
                "direction": TradeDirection.CALL,
                "metrics": {
                    "execute": True,
                    "deploy_ok": True,
                    "conviction": 0.55,
                    "trade_score": 0.55,
                    "val_accuracy": 0.40,
                    "edge": 0.05,
                    "raw_prob": 0.55,
                },
            },
        }
        orders = orch.executor._collect_orders(decisions)
        assert len(orders) == 1


def test_collect_orders_non_mandatory_keeps_filtered_candidate(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
        orch.config.setdefault("deep_learning", {})["min_edge_execute"] = 0.04
        orch.symbols = ["RDBULL"]
        orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["include_anchor_trades"] = True
        decisions = {
            "RDBULL": {
                "direction": TradeDirection.CALL,
                "metrics": {
                    "execute": True,
                    "deploy_ok": True,
                    "conviction": 0.80,
                    "trade_score": 0.80,
                    "val_accuracy": 0.70,
                    "edge": 0.10,
                    "raw_prob": 0.80,
                    "trend_direction": "CALL",
                    "indicators": {
                        "adx": 0.28,
                        "hurst": 0.55,
                        "vol_ratio": 1.1,
                        "rsi": 0.52,
                        "keltner": 0.55,
                        "cmo": 0.05,
                    },
                },
            },
        }
        orders = orch.executor._collect_orders(decisions)
        assert len(orders) == 1
        assert orders[0][1] == TradeDirection.CALL


@pytest.mark.asyncio
async def test_execute_cluster_mandatory_never_exec_skip_without_direction(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 3
        orch.symbols = ["RDBULL"]
        orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
        with (
            patch.object(orch.executor.logger, "warning") as mock_warn,
            patch.object(orch.executor, "_execute_orders", new_callable=AsyncMock, return_value=1) as mock_exec,
        ):
            await orch.executor.execute_cluster(
                {
                    "RDBULL": {
                        "direction": None,
                        "metrics": {"raw_prob": 0.58, "trade_score": 0.55, "val_accuracy": 0.55},
                    }
                }
            )
        assert not any("EXEC_SKIP" in str(c) for c in mock_warn.call_args_list)
        assert mock_exec.call_count == 1


@pytest.mark.asyncio
async def test_finalize_cluster_execution_without_orders_skips_idle_status_log(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 7
        with (
            patch.object(orch.executor.logger, "info") as mock_info,
            patch.object(orch.executor, "_collect_orders", return_value=[]),
            patch.object(orch.executor, "_execute_orders", return_value=0),
        ):
            await orch.executor.execute_cluster({})
        joined_info = "\n".join(str(c.args[0]) for c in mock_info.call_args_list if c.args)
        assert "STATUS: IDLE" not in joined_info


@pytest.mark.asyncio
async def test_execute_orders_uses_sixty_second_contract_duration(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 1
        orch.config.setdefault("risk_management", {}).setdefault("params", {})["duration"] = 60
        orch.config["risk_management"]["params"]["duration_unit"] = "s"
        orch.risk_manager.calculate_stake = MagicMock(return_value=2.0)
        place = AsyncMock(return_value=MagicMock(contract_id=99, payout=1.9, buy_price=2.0))
        with patch.object(orch.executor, "_place_order", place):
            await orch.executor._execute_orders(
                [("RDBULL", TradeDirection.CALL, {"conviction": 0.8})],
                0.0,
                1000.0,
            )
        assert place.await_args.kwargs["duration"] == 60
        assert place.await_args.kwargs["metrics"]["duration"] == 60


@pytest.mark.asyncio
async def test_execute_orders_market_closed_emits_warning(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 1
        orch.risk_manager.calculate_stake = MagicMock(return_value=2.0)
        orch.executor._place_order = AsyncMock(side_effect=Exception("This market is presently closed."))
        orders = [("RDBEAR", TradeDirection.CALL, {"conviction": 0.8})]
        with patch.object(orch.executor.logger, "warning") as mock_warn:
            count = await orch.executor._execute_orders(orders, 0.0, 100.0)
        assert count == 0
        mock_warn.assert_called_once()
        assert "SKIP" in mock_warn.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.real_settlement_watch
async def test_run_settlement_watch_schedules_cycle_and_handles_error(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.running = True
        orch.state.active_contracts = {}
        orch.schedule_trading_cycle_after_settlement = MagicMock()
        with patch(
            "src.application.services.orchestrator.execution_settlement.wait_for_settlement",
            new_callable=AsyncMock,
        ) as mock_wait:
            mock_wait.side_effect = RuntimeError("ws down")
            with patch.object(orch.executor.logger, "error") as mock_err:
                await orch.executor._run_settlement_watch()
        mock_err.assert_called_once()
        orch.schedule_trading_cycle_after_settlement.assert_called_once()
