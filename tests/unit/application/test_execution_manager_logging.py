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
