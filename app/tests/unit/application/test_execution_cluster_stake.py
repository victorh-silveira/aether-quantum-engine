from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import TradeDirection


def test_cluster_stake_block_passes_dl_metrics_to_stake_reason(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 7
        orch.risk_manager.pending_loss["OTC_SPC"] = 100.0
        orders = [
            (
                "OTC_SPC",
                TradeDirection.PUT,
                {
                    "conviction": 0.60,
                    "raw_prob": 0.58,
                    "val_brier": 0.12,
                    "deploy_ok": True,
                },
            ),
        ]
        with patch.object(
            orch.risk_manager, "stake_block_reason", wraps=orch.risk_manager.stake_block_reason
        ) as mock_reason:
            block = orch.executor._cluster_stake_block(orders, 5000.0)
        assert block is None
        mock_reason.assert_called_once()
        call_kw = mock_reason.call_args.kwargs
        assert call_kw["order_direction"] == "PUT"
        assert call_kw["dl_metrics"]["val_brier"] == 0.12
        assert call_kw["cycle_id"] == 7


@pytest.mark.asyncio
async def test_execute_cluster_logs_recovery_banner(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch._active_cycle_id = 12
        orch.risk_manager.pending_loss["OTC_SPC"] = 50.0
        orch.risk_manager.initial_bankroll = 10000.0
        with (
            patch.object(orch.executor.logger, "info") as mock_info,
            patch.object(orch.executor, "_collect_orders", return_value=[]),
            patch.object(orch.executor, "_execute_orders", new_callable=AsyncMock, return_value=0),
        ):
            await orch.executor.execute_cluster({})
        assert any("RISK: RECOVERY" in str(c) for c in mock_info.call_args_list)
