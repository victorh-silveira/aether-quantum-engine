"""Testes unitarios para protecoes de liquidação no ExecutionManager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.settlement_utils import clear_contract_metadata, clear_contract_tracking
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.state.trading_state import TradingState


@pytest.mark.asyncio
async def test_wait_for_settlement_prunes_orphan_contract_ids(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.risk_manager.active_contract_ids = [101]
        orch.risk_manager.contract_to_symbol[101] = "frxEURUSD"
        with (
            patch.object(orch.executor, "reconcile", AsyncMock()) as mock_reconcile,
            patch("src.application.services.orchestrator.execution_manager.asyncio.sleep", AsyncMock()),
        ):
            await orch.executor.wait_for_settlement(timeout=10)
        assert orch.risk_manager.active_contract_ids == []
        assert 101 not in orch.risk_manager.contract_to_symbol
        mock_reconcile.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_for_settlement_noop_when_empty(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        with patch.object(orch.executor, "reconcile", AsyncMock()) as mock_reconcile:
            await orch.executor.wait_for_settlement(timeout=1)
        assert orch.risk_manager.active_contract_ids == []
        mock_reconcile.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_for_settlement_clears_stagnant_pending_ids(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        ex = orch.config.setdefault("orchestrator", {}).setdefault("execution", {})
        ex["settlement_max_stagnant_polls"] = 2
        ex["settlement_stagnant_grace_seconds"] = 0.0
        ex["settlement_post_expiry_slack_seconds"] = 0.0
        orch.risk_manager.active_contract_ids = [202]
        orch.risk_manager.contract_to_symbol[202] = "OTC_SPC"
        await orch.state.add_contract(
            Contract(
                contract_id=202,
                proposal_id="p202",
                status=TradeStatus.OPEN,
                buy_price=40.0,
                payout=76.0,
                symbol="OTC_SPC",
                direction=TradeDirection.PUT,
                stake=40.0,
                expiry_time=1,
            )
        )
        with (
            patch.object(orch.executor, "reconcile", AsyncMock()),
            patch("src.application.services.orchestrator.execution_manager.asyncio.sleep", AsyncMock()),
        ):
            await orch.executor.wait_for_settlement(timeout=30)
        assert orch.risk_manager.active_contract_ids == []
        assert 202 not in orch.risk_manager.contract_to_symbol


def test_clear_contract_tracking_removes_ids_and_maps():
    risk = MagicMock()
    risk.active_contract_ids = [10, 11]
    risk.contract_to_symbol = {10: "OTC_FCHI", 11: "OTC_GDAXI"}
    risk.cluster_results = {10: "x", 11: "y"}
    clear_contract_tracking([10, 11], risk)
    assert risk.active_contract_ids == []
    assert risk.contract_to_symbol == {}
    assert risk.cluster_results == {}


def test_clear_contract_metadata_removes_only_maps():
    risk = MagicMock()
    risk.active_contract_ids = [10, 11]
    risk.contract_to_symbol = {10: "OTC_FCHI", 11: "OTC_GDAXI"}
    risk.cluster_results = {10: "x", 11: "y"}
    clear_contract_metadata([10, 11], risk)
    assert risk.active_contract_ids == [10, 11]
    assert risk.contract_to_symbol == {}
    assert risk.cluster_results == {}
