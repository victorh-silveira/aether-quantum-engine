"""Testes unitários para o módulo settlement_logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.settlement_logic import log_cluster_summary, process_contract_settlement
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
        symbol="R_75",
        direction=TradeDirection.PUT,
        stake=5.85,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [1694702639]
    orch.risk_manager.contract_to_symbol[1694702639] = "R_75"
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
        symbol="R_50",
        direction=TradeDirection.CALL,
        stake=10.0,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [123]
    orch.risk_manager.contract_to_symbol[123] = "R_50"
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
        symbol="R_75",
        direction=TradeDirection.CALL,
        stake=5.0,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [456]
    orch.risk_manager.contract_to_symbol[456] = "R_75"
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
    assert orch._last_loss_symbol == "R_75"
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
        symbol="R_50",
        direction=TradeDirection.CALL,
        stake=10.0,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [777]
    orch.risk_manager.contract_to_symbol[777] = "R_50"
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
        patch_instant_post_settlement_poll(),
    ):
        await process_contract_settlement(orch, data)

    assert pending_task.cancel.called
    assert orch.running is False
    assert orch.shutdown_reason == "stop_win"
    assert orch.risk_manager.total_session_profit == 150.0
