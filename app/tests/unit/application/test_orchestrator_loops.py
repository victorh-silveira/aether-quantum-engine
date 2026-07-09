from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.direction_persistence_guard import log_regime_guard, reset_regime_guard_log_state
from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.orchestrator.api_maintenance_guard import (
    _API_GUARD_LOG_MESSAGE,
    schedule_api_maintenance_hibernation,
)
from src.application.services.orchestrator.regime_freeze_yield import _REGIME_FREEZE_DEFAULT_YIELD_SECONDS
from src.application.services.orchestrator.session_persistence_barrier import (
    session_persistence_write_active,
)
from src.application.services.orchestrator.settlement_logic import process_contract_settlement
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from tests.unit.application.post_settlement_helpers import patch_instant_post_settlement_poll, strong_cycle_decisions


FREEZE_YIELD_MODULE = "src.application.services.orchestrator.regime_freeze_yield"
TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"


@pytest.mark.asyncio
async def test_settlement_loss_reconciles_planned_vs_executed_stake(orch_ready):
    orch = orch_ready
    orch._contract_cycle = {901: 4}
    contract = Contract(
        contract_id=901,
        proposal_id="p901",
        status=TradeStatus.OPEN,
        buy_price=332.28,
        payout=610.0,
        symbol="RDBULL",
        direction=TradeDirection.CALL,
        stake=390.92,
        expiry_time=0,
    )
    await orch.state.add_contract(contract)
    orch.risk_manager.active_contract_ids = [901]
    orch.risk_manager.contract_to_symbol[901] = "RDBULL"
    orch.risk_manager.contract_stakes[901] = 390.92
    orch.risk_manager.begin_cluster(1)
    data = {
        "proposal_open_contract": {
            "status": "lost",
            "is_settled": 1,
            "contract_id": 901,
            "buy_price": 332.28,
            "profit": -390.92,
            "balance_after": 10667.72,
        }
    }
    with patch_instant_post_settlement_poll():
        orch.executor.execute_cluster = AsyncMock()
        await process_contract_settlement(orch, data)
        if orch._post_settlement_task is not None:
            await orch._post_settlement_task
    assert orch.risk_manager.pending_loss.get("RDBULL") == pytest.approx(390.92)
    assert orch.risk_manager.last_loss_stake == pytest.approx(332.28)
    assert orch.risk_manager.total_session_profit == pytest.approx(-332.28)


def _frozen_decisions():
    return {
        "RDBULL": {
            "metrics": {
                "signal_status": SIGNAL_SUSPENDED,
                "execute": True,
                "trade_score": 0.70,
                "raw_prob": 0.70,
            }
        },
        "RDBEAR": {
            "metrics": {
                "signal_status": SIGNAL_SUSPENDED,
                "execute": True,
                "trade_score": 0.65,
                "raw_prob": 0.35,
            }
        },
    }


@pytest.mark.asyncio
async def test_run_trading_cycle_freeze_yields_and_avoids_hot_loop(orch_ready):
    orch = orch_ready
    orch._last_epoch = 120
    orch._last_cluster_cycle_end = 0.0
    orch._dl_fast_cycle = False
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    recorded: list[float] = []
    lock_during_sleep: list[bool] = []

    async def record_sleep(seconds: float) -> None:
        recorded.append(seconds)
        lock_during_sleep.append(orch.state_mgr._state_lock.locked())

    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=_frozen_decisions(),
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.refresh_correlation_cache", new_callable=AsyncMock),
        patch(f"{FREEZE_YIELD_MODULE}._yield_freeze_delay", side_effect=record_sleep),
    ):
        first = await run_trading_cycle_if_ready(orch)
        second = await run_trading_cycle_if_ready(orch)

    assert first is True
    assert second is False
    assert len(recorded) == 1
    assert recorded[0] == pytest.approx(_REGIME_FREEZE_DEFAULT_YIELD_SECONDS)
    assert lock_during_sleep == [False]
    assert not orch.state_mgr._state_lock.locked()


@pytest.mark.asyncio
async def test_run_trading_cycle_consecutive_freeze_releases_lock_before_sleep(orch_ready):
    orch = orch_ready
    orch._last_epoch = 120
    orch._last_cluster_cycle_end = 0.0
    orch._dl_fast_cycle = True
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    lock_during_sleep: list[bool] = []

    async def record_sleep(seconds: float) -> None:
        lock_during_sleep.append(orch.state_mgr._state_lock.locked())

    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=_frozen_decisions(),
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.refresh_correlation_cache", new_callable=AsyncMock),
        patch(f"{FREEZE_YIELD_MODULE}._yield_freeze_delay", side_effect=record_sleep),
    ):
        await run_trading_cycle_if_ready(orch)
        orch._last_epoch = 121
        await run_trading_cycle_if_ready(orch)

    assert len(lock_during_sleep) == 2
    assert lock_during_sleep == [False, False]
    assert not orch.state_mgr._state_lock.locked()


@pytest.mark.asyncio
async def test_run_trading_cycle_freeze_log_emitted_once_per_cycle_id(orch_ready, caplog):
    reset_regime_guard_log_state()
    orch = orch_ready
    orch._last_epoch = 120
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0

    async def noop_sleep(seconds: float) -> None:
        _ = seconds

    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=_frozen_decisions(),
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.refresh_correlation_cache", new_callable=AsyncMock),
        patch(f"{FREEZE_YIELD_MODULE}._yield_freeze_delay", side_effect=noop_sleep),
        caplog.at_level("INFO", logger="AETH"),
    ):
        log_regime_guard(1, "FREEZE: SKIP CYCLE", 2)
        log_regime_guard(1, "FREEZE: SKIP CYCLE", 2)
        await run_trading_cycle_if_ready(orch)

    freeze_logs = [record for record in caplog.records if "FREEZE: SKIP CYCLE" in record.message]
    assert len(freeze_logs) == 1


@pytest.mark.asyncio
async def test_run_trading_cycle_api_maintenance_hibernates_cleanly(orch_ready, caplog):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    schedule_api_maintenance_hibernation(
        orch,
        "Trading is not available from 00:00:00 to 00:01:00",
    )
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
        ) as collect_mock,
        caplog.at_level("INFO"),
    ):
        first = await run_trading_cycle_if_ready(orch)
        second = await run_trading_cycle_if_ready(orch)
    assert first is False
    assert second is False
    collect_mock.assert_not_awaited()
    guard_logs = [record for record in caplog.records if record.message == _API_GUARD_LOG_MESSAGE]
    assert len(guard_logs) == 1


@pytest.mark.asyncio
async def test_linear_reset_settlement_then_immediate_inference_runs_cleanly(orch_ready):
    orch = orch_ready
    loss_contract = Contract(
        contract_id=431,
        proposal_id="p431",
        status=TradeStatus.OPEN,
        buy_price=8.0,
        payout=14.0,
        symbol="RDBEAR",
        direction=TradeDirection.PUT,
        stake=8.0,
        expiry_time=0,
    )
    win_contract = Contract(
        contract_id=432,
        proposal_id="p432",
        status=TradeStatus.OPEN,
        buy_price=8.0,
        payout=15.0,
        symbol="RDBEAR",
        direction=TradeDirection.PUT,
        stake=8.0,
        expiry_time=0,
    )
    orch._contract_cycle = {431: 6, 432: 6}
    loss_data = {
        "proposal_open_contract": {
            "status": "lost",
            "is_settled": 1,
            "contract_id": 431,
            "profit": -8.0,
            "balance_after": 992.0,
        }
    }
    win_data = {
        "proposal_open_contract": {
            "status": "won",
            "is_settled": 1,
            "contract_id": 432,
            "profit": 10.0,
            "balance_after": 1002.0,
        }
    }
    await orch.state.add_contract(loss_contract)
    orch.risk_manager.active_contract_ids = [431]
    orch.risk_manager.contract_to_symbol[431] = "RDBEAR"
    orch.risk_manager.begin_cluster(1)
    with patch_instant_post_settlement_poll():
        await process_contract_settlement(orch, loss_data)
        if orch._post_settlement_task is not None:
            await orch._post_settlement_task

    await orch.state.add_contract(win_contract)
    orch.risk_manager.active_contract_ids = [432]
    orch.risk_manager.contract_to_symbol[432] = "RDBEAR"
    orch.risk_manager.begin_cluster(1)
    with (
        patch_instant_post_settlement_poll(),
        patch(
            "src.application.services.orchestrator.settlement_logic.run_linear_reset_persistence_barrier",
            new_callable=AsyncMock,
        ) as barrier_mock,
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=strong_cycle_decisions(),
        ) as collect_mock,
    ):
        orch.executor.execute_cluster = AsyncMock()
        await process_contract_settlement(orch, win_data)
        if orch._post_settlement_task is not None:
            await orch._post_settlement_task
    barrier_mock.assert_awaited_once()
    assert session_persistence_write_active(orch) is False
    collect_mock.assert_awaited_once()
    orch.executor.execute_cluster.assert_awaited_once()
