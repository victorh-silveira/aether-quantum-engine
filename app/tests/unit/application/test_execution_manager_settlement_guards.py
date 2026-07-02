"""Testes unitarios para protecoes de liquidação no ExecutionManager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.execution_settlement import (
    _next_stagnant_poll_count,
    _settlement_grace_period,
)
from src.application.services.orchestrator.settlement_utils import clear_contract_metadata, clear_contract_tracking
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from src.infrastructure.state.trading_state import TradingState
from tests.unit.application.post_settlement_helpers import (
    patch_instant_settlement_poll,
    patch_settlement_poll_clear_after,
)


def test_next_stagnant_poll_count_resets_during_grace():
    assert _next_stagnant_poll_count(3, 2.0, 10.0, [1], [1]) == 0


def test_settlement_grace_period_uses_max_of_dynamic_and_static():
    exec_mgr = MagicMock()
    exec_mgr.orch.config = {"risk_management": {"params": {"duration": 900, "duration_unit": "s"}}}
    exec_mgr.orch.state.active_contracts = []
    execution_cfg = {"settlement_post_expiry_slack_seconds": 2.0}
    with (
        patch(
            "src.application.services.orchestrator.execution_settlement.settlement_utils.min_elapsed_before_stagnant_polls",
            return_value=302.0,
        ),
        patch(
            "src.application.services.orchestrator.execution_settlement.settlement_utils.calculate_cluster_grace_period",
            return_value=21.0,
        ),
    ):
        grace = _settlement_grace_period(exec_mgr, execution_cfg, start_time=0.0)
    assert grace == 302.0


@pytest.mark.asyncio
async def test_wait_for_settlement_breaks_on_timeout(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.risk_manager.active_contract_ids = [404]
        with patch(
            "src.application.services.orchestrator.execution_settlement._settlement_timed_out",
            return_value=True,
        ) as mock_timed_out:
            await orch.executor.wait_for_settlement(timeout=1)
        mock_timed_out.assert_called()
        assert orch.risk_manager.active_contract_ids == [404]


@pytest.mark.asyncio
async def test_wait_for_settlement_prunes_orphan_contract_ids(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        orch.risk_manager.active_contract_ids = [101]
        orch.risk_manager.contract_to_symbol[101] = "RDBULL"
        with (
            patch.object(orch.executor, "reconcile", AsyncMock()) as mock_reconcile,
            patch_instant_settlement_poll(),
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
async def test_wait_for_settlement_keeps_stagnant_pending_ids(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        ex = orch.config.setdefault("orchestrator", {}).setdefault("execution", {})
        ex["settlement_max_stagnant_polls"] = 2
        ex["settlement_stagnant_grace_seconds"] = 0.0
        ex["settlement_post_expiry_slack_seconds"] = 0.0
        orch.risk_manager.active_contract_ids = [202]
        orch.risk_manager.contract_to_symbol[202] = "RDBEAR"
        await orch.state.add_contract(
            Contract(
                contract_id=202,
                proposal_id="p202",
                status=TradeStatus.OPEN,
                buy_price=40.0,
                payout=76.0,
                symbol="RDBEAR",
                direction=TradeDirection.PUT,
                stake=40.0,
                expiry_time=1,
            )
        )
        with (
            patch.object(orch.executor, "reconcile", AsyncMock(return_value=True)),
            patch(
                "src.application.services.orchestrator.execution_settlement.backfill_pending_contracts",
                AsyncMock(return_value=0),
            ),
            patch.object(orch, "_save_full_state", AsyncMock()),
            patch_settlement_poll_clear_after(orch.risk_manager, 8),
        ):
            await orch.executor.wait_for_settlement(timeout=5)
        assert 202 in orch.risk_manager.contract_to_symbol


@pytest.mark.asyncio
async def test_wait_for_settlement_backfill_recovers_before_clear(orch_config):
    TradingState.reset()
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
        ex = orch.config.setdefault("orchestrator", {}).setdefault("execution", {})
        ex["settlement_max_stagnant_polls"] = 1
        ex["settlement_stagnant_grace_seconds"] = 0.0
        ex["settlement_post_expiry_slack_seconds"] = 0.0
        orch.ws.is_running = True
        orch.risk_manager.active_contract_ids = [303]
        await orch.state.add_contract(
            Contract(
                contract_id=303,
                proposal_id="p",
                status=TradeStatus.OPEN,
                buy_price=2.0,
                payout=4.0,
                symbol="RDBEAR",
                direction=TradeDirection.CALL,
                stake=2.0,
                expiry_time=1,
            )
        )

        async def mock_backfill(_orch, _pending):
            orch.risk_manager.active_contract_ids = []
            return 1

        with (
            patch.object(orch.executor, "reconcile", AsyncMock(return_value=True)),
            patch(
                "src.application.services.orchestrator.execution_settlement.backfill_pending_contracts",
                side_effect=mock_backfill,
            ),
            patch_instant_settlement_poll(),
            patch.object(orch.logger, "info") as mock_info,
            patch.object(orch, "_save_full_state", AsyncMock()),
        ):
            await orch.executor.wait_for_settlement(timeout=5)
        assert orch.risk_manager.active_contract_ids == []
        assert any("Recuperados" in str(c) for c in mock_info.call_args_list)


def test_clear_contract_tracking_removes_ids_and_maps():
    risk = MagicMock()
    risk.active_contract_ids = [10, 11]
    risk.contract_to_symbol = {10: "RDBULL", 11: "RDBEAR"}
    risk.cluster_results = {10: "x", 11: "y"}
    clear_contract_tracking([10, 11], risk)
    assert risk.active_contract_ids == []
    assert risk.contract_to_symbol == {}
    assert risk.cluster_results == {}


def test_clear_contract_metadata_removes_only_maps():
    risk = MagicMock()
    risk.active_contract_ids = [10, 11]
    risk.contract_to_symbol = {10: "RDBULL", 11: "RDBEAR"}
    risk.cluster_results = {10: "x", 11: "y"}
    clear_contract_metadata([10, 11], risk)
    assert risk.active_contract_ids == [10, 11]
    assert risk.contract_to_symbol == {}
    assert risk.cluster_results == {}
