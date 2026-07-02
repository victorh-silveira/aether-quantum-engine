from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from tests.unit.application.post_settlement_helpers import (
    patch_instant_post_settlement_poll,
    patch_post_settlement_poll_stop_after,
)


def test_schedule_trading_cycle_early_returns(orch_ready):
    orch = orch_ready
    orch.running = False
    with patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.create_task") as mock_create:
        orch.schedule_trading_cycle_after_settlement()
    mock_create.assert_not_called()

    orch.running = True
    orch.state.active_contracts = {1: object()}
    with patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.create_task") as mock_create:
        orch.schedule_trading_cycle_after_settlement()
    mock_create.assert_not_called()

    orch.state.active_contracts = {}
    orch._post_settlement_task = None
    pending = MagicMock()
    pending.done.return_value = False
    orch._post_settlement_task = pending
    orch._post_settlement_wake.clear()
    with patch("src.application.services.orchestrator.post_settlement_cycle.asyncio.create_task") as mock_create:
        orch.schedule_trading_cycle_after_settlement()
    mock_create.assert_not_called()
    assert orch._post_settlement_wake.is_set()


@pytest.mark.asyncio
async def test_schedule_trading_cycle_after_settlement_runs_real_cycle(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_instant_post_settlement_poll(),
    ):
        orch.executor.execute_cluster = AsyncMock()
        orch.schedule_trading_cycle_after_settlement()
        await orch._post_settlement_task
    orch.executor.execute_cluster.assert_awaited_once()


@pytest.mark.asyncio
async def test_post_settlement_breath_runs_real_cycle_when_running(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0.01
    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "src.application.services.orchestrator.post_settlement_cycle._await_post_settlement_breath",
            new_callable=AsyncMock,
        ) as mock_breath,
    ):
        orch.executor.execute_cluster = AsyncMock()
        await orch._run_post_settlement_breath_and_cycle()
        mock_breath.assert_awaited_once()
    orch.executor.execute_cluster.assert_awaited_once()

    orch.running = False
    with patch_instant_post_settlement_poll():
        orch.executor.execute_cluster.reset_mock()
        await orch._run_post_settlement_breath_and_cycle()
    orch.executor.execute_cluster.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_settlement_breath_skips_when_active_contracts(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.state.active_contracts = {1: object()}

    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_post_settlement_poll_stop_after(orch, 3),
    ):
        orch.executor.execute_cluster = AsyncMock()
        await orch._run_post_settlement_breath_and_cycle()
    orch.executor.execute_cluster.assert_not_awaited()


@pytest.mark.asyncio
async def test_settlement_win_triggers_real_next_cycle(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch._contract_cycle = {1692883719: 1}
    contract = Contract(
        contract_id=1692883719,
        proposal_id="p1",
        status=TradeStatus.OPEN,
        buy_price=5.83,
        payout=10.60,
        symbol="RDBEAR",
        direction=TradeDirection.PUT,
        stake=5.83,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [1692883719]
    orch.risk_manager.contract_to_symbol[1692883719] = "RDBEAR"
    orch.risk_manager.begin_cluster(1)

    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_instant_post_settlement_poll(),
    ):
        orch.executor.execute_cluster = AsyncMock()
        await orch._on_contract_update(
            {
                "proposal_open_contract": {
                    "contract_id": 1692883719,
                    "is_settled": 1,
                    "status": "won",
                    "profit": 4.77,
                    "balance_after": 1004.77,
                }
            }
        )
        assert orch._post_settlement_task is not None
        await orch._post_settlement_task

    assert 1692883719 not in orch.state.active_contracts
    assert 1692883719 not in orch.risk_manager.active_contract_ids
    orch.executor.execute_cluster.assert_awaited_once()


@pytest.mark.asyncio
async def test_settlement_loss_triggers_real_next_cycle(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch._contract_cycle = {1694702639: 1}
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

    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_instant_post_settlement_poll(),
    ):
        orch.executor.execute_cluster = AsyncMock()
        await orch._on_contract_update(
            {
                "proposal_open_contract": {
                    "contract_id": 1694702639,
                    "is_settled": 1,
                    "status": "lost",
                    "profit": -5.85,
                    "balance_after": 994.15,
                }
            }
        )
        assert orch._post_settlement_task is not None
        await orch._post_settlement_task

    assert 1694702639 not in orch.state.active_contracts
    orch.executor.execute_cluster.assert_awaited_once()
