from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_execution_manager_log_line_contains_exec_and_direction(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        with patch.object(orch.executor.logger, "debug") as mock_dbg:
            orch.executor._log_exec(
                "OTC_FCHI",
                TradeDirection.CALL,
                1.0,
                {"conviction": 1.0},
                order_n=1,
                contract_id=12345,
            )
        args = mock_dbg.call_args_list[0].args
        assert "ORDEM ENVIADA" in str(args[0])
        assert args[5] == "12345"


def test_execution_manager_collect_orders_keeps_execute_false_if_forced_in_dict(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.symbols = ["frxEURUSD", "OTC_SPC", "OTC_GDAXI"]
        decisions = {
            "frxEURUSD": {"direction": TradeDirection.CALL, "metrics": {"conviction": 0.8, "execute": False}},
            "OTC_SPC": {"direction": TradeDirection.PUT, "metrics": {"conviction": 0.9, "execute": True}},
        }
        orders = orch.executor._collect_orders(decisions, include_anchor=True)
        assert len(orders) == 1
        assert orders[0][0] == "OTC_SPC"


@pytest.mark.asyncio
async def test_execute_cluster_logs_exec_pause_on_stop_win(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 2
        orch.symbols = ["frxEURUSD", "OTC_FCHI"]
        orch.risk_manager.stake_block_reason = MagicMock(return_value="stop_win")
        decisions = {
            "OTC_FCHI": {"direction": TradeDirection.PUT, "metrics": {"conviction": 0.7, "execute": True}},
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
        orders = [("OTC_FCHI", TradeDirection.CALL, {"conviction": 0.7})]
        count = await orch.executor._execute_orders(orders, 0.0, 49.0)
        assert count == 0


def test_cluster_stake_block_empty_orders(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        assert orch.executor._cluster_stake_block([], 50.0) is None


def test_log_execution_blockers_stake_zero(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 4
        orch.symbols = ["frxEURUSD", "OTC_FCHI"]
        orch.risk_manager.calculate_stake = MagicMock(return_value=0.0)
        orch.risk_manager.stake_block_reason = MagicMock(return_value="kelly_no_edge")
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {"OTC_FCHI": {"direction": TradeDirection.PUT, "metrics": {"conviction": 0.7, "execute": True}}},
                include_anchor=False,
            )
        assert "kelly_no_edge" in mock_info.call_args.args[2]


def test_log_execution_blockers_sem_direcao(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 3
        orch.symbols = ["frxEURUSD", "OTC_FCHI"]
        with patch.object(orch.executor.logger, "info") as mock_info:
            orch.executor._log_execution_blockers(
                {"OTC_FCHI": {"direction": None, "metrics": {"execute": True}}},
                include_anchor=False,
            )
        assert "sem_direcao" in mock_info.call_args.args[2]


@pytest.mark.asyncio
async def test_execute_cluster_skips_on_refresh_without_llm(orch_config):
    orch_config.setdefault("orchestrator", {})["cluster_refresh_execute_enabled"] = False
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 4
        orch._cluster_refresh_without_llm = True
        orch.symbols = ["frxEURUSD", "OTC_FCHI"]
        decisions = {
            "frxEURUSD": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "risk_off"}},
            "OTC_FCHI": {
                "direction": TradeDirection.CALL,
                "metrics": {"conviction": 0.7, "execute": True, "macro_sentiment": "risk_off"},
            },
        }
        with (
            patch.object(orch.executor.logger, "info") as mock_info,
            patch.object(orch.executor, "_execute_orders", new_callable=AsyncMock) as mock_exec,
        ):
            await orch.executor.execute_cluster(decisions)
        mock_exec.assert_not_awaited()
        assert any(
            "EXEC_SKIP" in str(c) and "divergence_refresh_no_quant_edge" in str(c) for c in mock_info.call_args_list
        )


@pytest.mark.asyncio
async def test_execute_cluster_allows_refresh_divergence_quant_validated(orch_config):
    orch_config.setdefault("orchestrator", {})["cluster_refresh_execute_enabled"] = False
    orch_config.setdefault("orchestrator", {})["cluster_refresh_execute_on_quant_validate"] = True
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 5
        orch._cluster_refresh_without_llm = True
        orch.symbols = ["frxEURUSD", "OTC_DJI"]
        orch.risk_manager.calculate_stake = MagicMock(return_value=2.0)
        decisions = {
            "frxEURUSD": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "divergence_us_leads"}},
            "OTC_DJI": {
                "direction": TradeDirection.PUT,
                "metrics": {
                    "conviction": 0.7,
                    "execute": True,
                    "macro_sentiment": "divergence_us_leads",
                    "llm_statarb_dir_corrected": True,
                    "cluster_target_sym": "OTC_DJI",
                },
            },
        }
        with (
            patch.object(orch.executor.logger, "info"),
            patch.object(orch.executor, "_execute_orders", new_callable=AsyncMock, return_value=1) as mock_exec,
        ):
            await orch.executor.execute_cluster(decisions)
        mock_exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_cluster_logs_exec_none_when_execute_false(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 1
        orch.symbols = ["frxEURUSD", "OTC_FCHI"]
        decisions = {
            "OTC_FCHI": {"direction": TradeDirection.PUT, "metrics": {"conviction": 0.7, "execute": False}},
        }
        with (
            patch.object(orch.executor.logger, "info") as mock_info,
            patch.object(orch.executor, "_execute_orders", new_callable=AsyncMock, return_value=0),
        ):
            await orch.executor.execute_cluster(decisions)
        assert any("EXEC_NONE" in str(c) for c in mock_info.call_args_list)


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
async def test_execute_orders_market_closed_emits_warning(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 1
        orch.risk_manager.calculate_stake = MagicMock(return_value=2.0)
        orch.executor._place_order = AsyncMock(side_effect=Exception("This market is presently closed."))
        orders = [("OTC_SSMI", TradeDirection.CALL, {"conviction": 0.8})]
        with patch.object(orch.executor.logger, "warning") as mock_warn:
            count = await orch.executor._execute_orders(orders, 0.0, 100.0)
        assert count == 0
        mock_warn.assert_called_once()
        assert "SKIP" in mock_warn.call_args.args[0]


@pytest.mark.asyncio
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
