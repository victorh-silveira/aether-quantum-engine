"""Testes unitários para o módulo settlement_logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.settlement_logic import (
    check_session_limits_before_post_settlement,
    log_cluster_summary,
    process_contract_settlement,
)
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from tests.unit.application.post_settlement_helpers import patch_instant_post_settlement_poll


@pytest.mark.asyncio
async def test_process_contract_settlement_ignores_open_contract(orch_ready):
    await process_contract_settlement(orch_ready, {"proposal_open_contract": {"status": "open", "contract_id": 1}})
    assert orch_ready.state.active_contracts == {}


@pytest.mark.asyncio
async def test_process_contract_settlement_ignores_premature_open_with_is_settled(orch_ready):
    orch = orch_ready
    contract = Contract(
        contract_id=1694702639,
        proposal_id="p1",
        status=TradeStatus.OPEN,
        buy_price=5.85,
        payout=10.63,
        symbol="RDBEAR",
        direction=TradeDirection.PUT,
        stake=5.85,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [1694702639]
    orch.risk_manager.contract_to_symbol[1694702639] = "RDBEAR"
    orch.risk_manager.begin_cluster(1)

    with patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.create_task") as mock_create:
        await process_contract_settlement(
            orch,
            {
                "proposal_open_contract": {
                    "contract_id": 1694702639,
                    "is_settled": 1,
                    "status": "open",
                    "profit": -5.85,
                }
            },
        )
        mock_create.assert_not_called()

    assert 1694702639 in orch.state.active_contracts
    assert 1694702639 in orch.risk_manager.active_contract_ids


@pytest.mark.asyncio
async def test_process_contract_settlement_won(orch_ready):
    orch = orch_ready
    orch.config.setdefault("risk_management", {})["large_account_stop_win_pct"] = 4.0
    orch._contract_cycle = {123: 1}
    contract = Contract(
        contract_id=123,
        proposal_id="p1",
        status=TradeStatus.OPEN,
        buy_price=10.0,
        payout=18.0,
        symbol="RDBULL",
        direction=TradeDirection.CALL,
        stake=10.0,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [123]
    orch.risk_manager.contract_to_symbol[123] = "RDBULL"
    orch.risk_manager.begin_cluster(1)

    data = {
        "proposal_open_contract": {
            "status": "won",
            "is_settled": 1,
            "contract_id": 123,
            "profit": 10.0,
            "balance_after": 1010.0,
        }
    }

    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_instant_post_settlement_poll(),
    ):
        orch.executor.execute_cluster = AsyncMock()
        await process_contract_settlement(orch, data)
        if orch._post_settlement_task is not None:
            await orch._post_settlement_task

    assert orch.state.balance == 1010.0
    assert orch._session_wins == 1
    assert 123 not in orch.state.active_contracts
    assert 123 not in orch.risk_manager.active_contract_ids
    assert orch._post_settlement_task is None
    orch.executor.execute_cluster.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_contract_settlement_lost(orch_ready):
    orch = orch_ready
    orch._contract_cycle = {456: 2}
    orch._buffer_result_logs = True
    orch._pending_result_logs = []
    contract = Contract(
        contract_id=456,
        proposal_id="p2",
        status=TradeStatus.OPEN,
        buy_price=5.0,
        payout=9.0,
        symbol="RDBEAR",
        direction=TradeDirection.CALL,
        stake=5.0,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [456]
    orch.risk_manager.contract_to_symbol[456] = "RDBEAR"
    orch.risk_manager.begin_cluster(1)
    data = {
        "proposal_open_contract": {
            "status": "lost",
            "is_settled": 1,
            "contract_id": 456,
            "profit": -5.0,
            "balance_after": 995.0,
        }
    }

    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_instant_post_settlement_poll(),
    ):
        orch.executor.execute_cluster = AsyncMock()
        await process_contract_settlement(orch, data)
        if orch._post_settlement_task is not None:
            await orch._post_settlement_task

    assert orch.state.balance == 995.0
    assert orch._session_losses == 1
    assert orch._last_loss_symbol == "RDBEAR"
    assert orch._last_loss_direction == "CALL"
    assert len(orch._pending_result_logs) == 1
    assert 456 not in orch.risk_manager.active_contract_ids
    orch.executor.execute_cluster.assert_awaited_once()


def test_log_cluster_summary(orch_ready):
    orch = orch_ready
    orch._last_result_cycle_id = 1
    orch.state.balance = 1050.0
    orch._session_wins = 2
    orch._session_losses = 1
    orch._cluster_results = [{"some": "data"}]

    log_cluster_summary(orch)

    assert orch._cluster_results == []


@pytest.mark.asyncio
async def test_process_contract_settlement_early_exits(orch_ready):
    orch = orch_ready

    await process_contract_settlement(orch, {})
    assert orch.state.active_contracts == {}

    await process_contract_settlement(orch, {"proposal_open_contract": {"is_settled": 1}})
    assert orch.state.active_contracts == {}

    await process_contract_settlement(orch, {"proposal_open_contract": {"is_settled": 1, "contract_id": 999}})
    assert orch._session_wins == 0
    assert orch._session_losses == 0


@pytest.mark.asyncio
async def test_process_contract_settlement_stop_win(orch_ready):
    orch = orch_ready
    orch._contract_cycle = {777: 1}
    orch.config["risk_management"] = {"stop_win_percentage": 5.0}
    contract = Contract(
        contract_id=777,
        proposal_id="p3",
        status=TradeStatus.OPEN,
        buy_price=10.0,
        payout=18.0,
        symbol="RDBULL",
        direction=TradeDirection.CALL,
        stake=10.0,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [777]
    orch.risk_manager.contract_to_symbol[777] = "RDBULL"
    orch.risk_manager.begin_cluster(1)
    orch.risk_manager.initial_bankroll = 1000.0
    orch.risk_manager.total_session_profit = 50.0

    data = {
        "proposal_open_contract": {
            "status": "won",
            "is_settled": 1,
            "contract_id": 777,
            "profit": 100.0,
            "balance_after": 1100.0,
        }
    }

    pending_task = MagicMock()
    pending_task.done.return_value = False
    orch._post_settlement_task = pending_task
    with (
        patch("src.application.services.orchestrator.settlement_logic.resolve_stop_win_target", return_value=50.0),
        patch(
            "src.application.services.orchestrator.settlement_logic.graceful_shutdown",
            new_callable=AsyncMock,
        ) as shutdown_mock,
        patch_instant_post_settlement_poll(),
    ):
        await process_contract_settlement(orch, data)

    shutdown_mock.assert_awaited_once()
    assert shutdown_mock.await_args.kwargs["fast_path"] is True
    assert orch.shutdown_reason == "stop_win"
    assert orch.risk_manager.total_session_profit == 150.0


def test_check_session_limits_before_post_settlement_detects_stop_win(orch_ready):
    orch = orch_ready
    orch.state.balance = 1060.0
    orch.state_mgr.reset_session_metrics(1000.0, 50.0)
    orch.state_mgr.state.total_trades_today = 2
    orch.risk_manager.total_session_profit = 60.0
    assert check_session_limits_before_post_settlement(orch) is True
    assert orch.state_mgr.state.stop_win_triggered is True


def test_check_session_limits_without_state_manager_returns_pnl_fallback(orch_ready):
    orch = orch_ready
    orch.state_mgr = MagicMock()
    orch.risk_manager.total_session_profit = 60.0
    with patch(
        "src.application.services.orchestrator.settlement_logic.resolve_stop_win_target",
        return_value=50.0,
    ):
        assert check_session_limits_before_post_settlement(orch) is True


def test_check_session_limits_triggers_on_session_pnl_before_state_sync(orch_ready):
    orch = orch_ready
    orch.risk_manager.total_session_profit = 105.09
    orch.state_mgr.reset_session_metrics(1000.0, 101.83)
    orch.state.balance = 1105.09
    with patch(
        "src.application.services.orchestrator.settlement_logic.resolve_stop_win_target",
        return_value=101.83,
    ):
        assert check_session_limits_before_post_settlement(orch) is True


def test_check_session_limits_via_balance_sync_when_pnl_below_target(orch_ready):
    orch = orch_ready
    orch.risk_manager.total_session_profit = 40.0
    orch.state.balance = 1060.0
    orch.state_mgr.reset_session_metrics(1000.0, 50.0)
    orch.state_mgr.state.total_trades_today = 2
    with patch(
        "src.application.services.orchestrator.settlement_logic.resolve_stop_win_target",
        return_value=50.0,
    ):
        assert check_session_limits_before_post_settlement(orch) is True
