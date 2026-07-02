import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from tests.unit.application.post_settlement_helpers import patch_instant_post_settlement_poll


@pytest.mark.asyncio
async def test_idle_watchdog_runs_cycle_when_idle(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["idle_cycle_watchdog_seconds"] = 0.01
    orch._last_idle_watchdog_attempt = 0.0
    with patch(
        "src.application.services.orchestrator.collect_deep_learning_decisions",
        new_callable=AsyncMock,
        return_value={},
    ):
        orch.executor.execute_cluster = AsyncMock()
        await orch._tick_idle_cycle_watchdog()
    orch.executor.execute_cluster.assert_awaited_once()


@pytest.mark.asyncio
async def test_idle_watchdog_skips_when_post_settlement_pending(orch_ready):
    orch = orch_ready
    pending = MagicMock()
    pending.done.return_value = False
    orch._post_settlement_task = pending
    with patch(
        "src.application.services.orchestrator.collect_deep_learning_decisions",
        new_callable=AsyncMock,
        return_value={},
    ):
        orch.executor.execute_cluster = AsyncMock()
        await orch._tick_idle_cycle_watchdog()
    orch.executor.execute_cluster.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_watchdog_skips_when_disabled_or_too_soon(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["idle_cycle_watchdog_seconds"] = 0
    with patch(
        "src.application.services.orchestrator.collect_deep_learning_decisions",
        new_callable=AsyncMock,
        return_value={},
    ):
        orch.executor.execute_cluster = AsyncMock()
        await orch._tick_idle_cycle_watchdog()
    orch.executor.execute_cluster.assert_not_awaited()

    orch.config["orchestrator"]["idle_cycle_watchdog_seconds"] = 60
    orch._last_idle_watchdog_attempt = time.time()
    await orch._tick_idle_cycle_watchdog()
    orch.executor.execute_cluster.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_candle_skips_when_post_settlement_pending(orch_ready):
    orch = orch_ready
    pending = MagicMock()
    pending.done.return_value = False
    orch._post_settlement_task = pending
    candle = MagicMock(symbol=orch.anchor, epoch=orch._last_epoch + 60)
    with patch(
        "src.application.services.orchestrator.collect_deep_learning_decisions",
        new_callable=AsyncMock,
        return_value={},
    ):
        orch.executor.execute_cluster = AsyncMock()
        await orch._on_candle(candle)
    orch.executor.execute_cluster.assert_not_awaited()


@pytest.mark.asyncio
async def test_settlement_schedules_cycle_before_save(orch_ready):
    orch = orch_ready
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
    call_order: list[str] = []
    original_schedule = orch.schedule_trading_cycle_after_settlement

    def track_schedule():
        call_order.append("schedule")
        original_schedule()

    orch.schedule_trading_cycle_after_settlement = track_schedule
    save_mock = AsyncMock(side_effect=lambda: call_order.append("save"))
    orch._save_full_state = save_mock
    with patch_instant_post_settlement_poll():
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
    assert call_order == ["schedule", "save"]


@pytest.mark.asyncio
async def test_stop_cancels_pending_post_settlement_task(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as mock_ws_class:
        mock_ws = mock_ws_class.return_value
        mock_ws.close = AsyncMock()
        orch = Orchestrator(orch_config, "token")
        pending = MagicMock()
        pending.done.return_value = False
        orch._post_settlement_task = pending
        deferred = MagicMock()
        deferred.done.return_value = False
        orch._dl_deferred_tasks = {"RDBEAR": deferred}
        await orch.stop()
        pending.cancel.assert_called_once()
        deferred.cancel.assert_called_once()
