from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.session_persistence_barrier import (
    _LINEAR_RESET_YIELD_SECONDS,
    _persist_session_state_snapshot,
    consume_linear_reset_flag,
    linear_reset_occurred,
    run_linear_reset_persistence_barrier,
    session_persistence_blocks_trading_cycle,
    session_persistence_write_active,
)
from src.application.services.orchestrator.settlement_logic import process_contract_settlement
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from tests.unit.application.post_settlement_helpers import patch_instant_post_settlement_poll


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"
BARRIER_MODULE = "src.application.services.orchestrator.session_persistence_barrier"


def test_linear_reset_flag_lifecycle(orch_ready):
    orch = orch_ready
    orch.risk_manager._linear_reset_occurred = True
    assert linear_reset_occurred(orch.risk_manager) is True
    assert consume_linear_reset_flag(orch) is True
    assert linear_reset_occurred(orch.risk_manager) is False
    assert consume_linear_reset_flag(orch) is False


def test_persist_session_state_snapshot_legacy_state_manager_branch(orch_ready):
    class StateManagerLegacy:
        def __init__(self):
            self.state = SimpleNamespace(
                current_balance=0.0,
                initial_balance=0.0,
                daily_stop_win_target=0.0,
            )

        def check_session_limits(self):
            return None

        def save_state(self):
            return None

    StateManagerLegacy.__name__ = "StateManager"
    orch = orch_ready
    legacy_mgr = StateManagerLegacy()
    orch.state_mgr = legacy_mgr
    orch.state.balance = 1234.5
    orch.risk_manager.initial_bankroll = 1000.0
    _persist_session_state_snapshot(orch)
    assert legacy_mgr.state.current_balance == pytest.approx(1234.5)
    assert legacy_mgr.state.initial_balance == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_run_linear_reset_persistence_barrier_sequences_save_and_yield(orch_ready):
    orch = orch_ready
    orch.risk_manager.consecutive_losses_linear = 3
    orch.risk_manager.last_loss_stake = 12.0
    orch.risk_manager.pending_loss = {"RDBULL": 0.0}
    orch._persist_full_state_unlocked = AsyncMock()
    with patch(f"{BARRIER_MODULE}.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
        await run_linear_reset_persistence_barrier(orch)
    assert orch.risk_manager.consecutive_losses_linear == 0
    assert orch.risk_manager.last_loss_stake == 0.0
    assert "RDBULL" not in orch.risk_manager.pending_loss
    orch._persist_full_state_unlocked.assert_awaited_once()
    assert sleep_mock.await_count == 2
    assert sleep_mock.await_args_list[-1].args[0] == pytest.approx(_LINEAR_RESET_YIELD_SECONDS)
    assert session_persistence_write_active(orch) is False


@pytest.mark.asyncio
async def test_run_linear_reset_persistence_barrier_without_state_manager(orch_ready):
    orch = orch_ready
    orch.state_mgr = MagicMock()
    orch._persist_full_state_unlocked = AsyncMock()
    with patch(f"{BARRIER_MODULE}.asyncio.sleep", new_callable=AsyncMock):
        await run_linear_reset_persistence_barrier(orch)
    orch._persist_full_state_unlocked.assert_awaited_once()


@pytest.mark.asyncio
async def test_trading_cycle_skips_execute_cluster_when_persistence_lock_races(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0

    async def _collect_and_lock(_orch):
        _orch._session_persistence_write_active = True
        return {"RDBULL": {"direction": None, "metrics": {"execute": False}}}

    with patch(
        f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
        side_effect=_collect_and_lock,
    ):
        orch.executor.execute_cluster = AsyncMock()
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is True
    orch.executor.execute_cluster.assert_not_awaited()
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch._session_persistence_write_active = True
    with patch(
        f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
        new_callable=AsyncMock,
    ) as collect_mock:
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is False
    collect_mock.assert_not_awaited()
    assert session_persistence_blocks_trading_cycle(orch) is True


@pytest.mark.asyncio
async def test_run_linear_reset_persistence_barrier_yield_seconds(orch_ready):
    orch = orch_ready
    orch._save_full_state = AsyncMock()
    with patch(f"{BARRIER_MODULE}.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
        await run_linear_reset_persistence_barrier(orch)
    assert sleep_mock.await_args_list[-1].args[0] == pytest.approx(_LINEAR_RESET_YIELD_SECONDS)


@pytest.mark.asyncio
async def test_process_contract_settlement_linear_reset_runs_persistence_barrier(orch_ready):
    orch = orch_ready
    loss_contract = Contract(
        contract_id=320,
        proposal_id="p320",
        status=TradeStatus.OPEN,
        buy_price=10.0,
        payout=18.0,
        symbol="RDBULL",
        direction=TradeDirection.CALL,
        stake=10.0,
        expiry_time=0,
    )
    win_contract = Contract(
        contract_id=321,
        proposal_id="p321",
        status=TradeStatus.OPEN,
        buy_price=10.0,
        payout=20.0,
        symbol="RDBULL",
        direction=TradeDirection.CALL,
        stake=10.0,
        expiry_time=0,
    )
    orch._contract_cycle = {320: 5, 321: 5}
    loss_data = {
        "proposal_open_contract": {
            "status": "lost",
            "is_settled": 1,
            "contract_id": 320,
            "profit": -10.0,
            "balance_after": 990.0,
        }
    }
    win_data = {
        "proposal_open_contract": {
            "status": "won",
            "is_settled": 1,
            "contract_id": 321,
            "profit": 12.0,
            "balance_after": 1002.0,
        }
    }
    await orch.state.add_contract(loss_contract)
    orch.risk_manager.active_contract_ids = [320]
    orch.risk_manager.contract_to_symbol[320] = "RDBULL"
    orch.risk_manager.begin_cluster(1)
    with patch_instant_post_settlement_poll():
        await process_contract_settlement(orch, loss_data)
        if orch._post_settlement_task is not None:
            await orch._post_settlement_task
    assert orch.risk_manager.consecutive_losses_linear == 1
    assert orch.risk_manager.pending_loss["RDBULL"] == pytest.approx(10.0)

    await orch.state.add_contract(win_contract)
    orch.risk_manager.active_contract_ids = [321]
    orch.risk_manager.contract_to_symbol[321] = "RDBULL"
    orch.risk_manager.begin_cluster(1)
    with (
        patch(
            "src.application.services.orchestrator.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_instant_post_settlement_poll(),
        patch(
            "src.application.services.orchestrator.settlement_logic.run_linear_reset_persistence_barrier",
            new_callable=AsyncMock,
        ) as barrier_mock,
    ):
        orch.executor.execute_cluster = AsyncMock()
        await process_contract_settlement(orch, win_data)
        if orch._post_settlement_task is not None:
            await orch._post_settlement_task
    barrier_mock.assert_awaited_once()
    assert orch.risk_manager.consecutive_losses_linear == 0
    assert session_persistence_write_active(orch) is False
    orch.executor.execute_cluster.assert_awaited_once()
