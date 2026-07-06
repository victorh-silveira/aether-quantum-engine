from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.post_settlement_cycle import run_post_settlement_breath_and_cycle
from src.application.services.orchestrator.post_settlement_loss_cooldown import (
    POST_LOSS_COOLDOWN_BASE_SECONDS,
    POST_LOSS_COOLDOWN_GROWTH,
    await_post_loss_cooldown,
    post_loss_cooldown_active,
    post_loss_cooldown_delay_seconds,
)
from src.application.services.orchestrator.settlement_logic import process_contract_settlement
from src.domain.models.trade import Contract, TradeDirection, TradeStatus
from tests.unit.application.post_settlement_helpers import patch_instant_post_settlement_poll


COOLDOWN_MODULE = "src.application.services.orchestrator.post_settlement_loss_cooldown"


def test_post_loss_cooldown_delay_zero_below_linear_two():
    assert post_loss_cooldown_delay_seconds(0) == 0.0
    assert post_loss_cooldown_delay_seconds(1) == 0.0


def test_post_loss_cooldown_delay_exponential_from_linear_two():
    assert post_loss_cooldown_delay_seconds(2) == pytest.approx(
        POST_LOSS_COOLDOWN_BASE_SECONDS * POST_LOSS_COOLDOWN_GROWTH**2
    )
    assert post_loss_cooldown_delay_seconds(3) == pytest.approx(
        POST_LOSS_COOLDOWN_BASE_SECONDS * POST_LOSS_COOLDOWN_GROWTH**3
    )
    assert post_loss_cooldown_delay_seconds(4) == pytest.approx(
        POST_LOSS_COOLDOWN_BASE_SECONDS * POST_LOSS_COOLDOWN_GROWTH**4
    )


def test_post_loss_cooldown_active_requires_loss_and_linear_floor():
    assert post_loss_cooldown_active("LOSS", 2) is True
    assert post_loss_cooldown_active("loss", 3) is True
    assert post_loss_cooldown_active("LOSS", 1) is False
    assert post_loss_cooldown_active("WIN", 4) is False
    assert post_loss_cooldown_active("FLAT", 4) is False


@pytest.mark.asyncio
async def test_await_post_loss_cooldown_level_three_virtual_clock(orch_ready):
    orch = orch_ready
    orch._last_settlement_outcome = "LOSS"
    orch.risk_manager.consecutive_losses_linear = 3
    expected = post_loss_cooldown_delay_seconds(3)
    recorded: list[float] = []

    async def record_sleep(seconds: float) -> None:
        recorded.append(seconds)

    with patch(f"{COOLDOWN_MODULE}.asyncio.sleep", side_effect=record_sleep):
        delay = await await_post_loss_cooldown(orch)
    assert delay == pytest.approx(expected)
    assert recorded == [pytest.approx(expected)]
    assert expected == pytest.approx(36.905625)


@pytest.mark.asyncio
async def test_await_post_loss_cooldown_skips_when_orchestrator_stopped(orch_ready):
    orch = orch_ready
    orch.running = False
    orch._last_settlement_outcome = "LOSS"
    orch.risk_manager.consecutive_losses_linear = 3
    with patch(f"{COOLDOWN_MODULE}.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
        delay = await await_post_loss_cooldown(orch)
    assert delay == 0.0
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_await_post_loss_cooldown_skips_on_win(orch_ready):
    orch = orch_ready
    orch._last_settlement_outcome = "WIN"
    orch.risk_manager.consecutive_losses_linear = 4
    recorded: list[float] = []

    async def record_sleep(seconds: float) -> None:
        recorded.append(seconds)

    with patch(f"{COOLDOWN_MODULE}.asyncio.sleep", side_effect=record_sleep):
        delay = await await_post_loss_cooldown(orch)
    assert delay == 0.0
    assert recorded == []


@pytest.mark.asyncio
async def test_sequential_loss_levels_expand_post_settlement_cooldown(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch._run_trading_cycle_if_ready = AsyncMock(return_value=True)
    delays_by_linear = {}

    async def record_sleep(seconds: float) -> None:
        linear = int(orch.risk_manager.consecutive_losses_linear)
        delays_by_linear[linear] = seconds

    for linear in (2, 3, 4):
        orch._last_settlement_outcome = "LOSS"
        orch.risk_manager.consecutive_losses_linear = linear
        with (
            patch(f"{COOLDOWN_MODULE}.asyncio.sleep", side_effect=record_sleep),
            patch_instant_post_settlement_poll(),
        ):
            await run_post_settlement_breath_and_cycle(orch)
        assert delays_by_linear[linear] == pytest.approx(post_loss_cooldown_delay_seconds(linear))

    assert delays_by_linear[3] > delays_by_linear[2]
    assert delays_by_linear[4] > delays_by_linear[3]


@pytest.mark.asyncio
async def test_run_post_settlement_skips_cooldown_when_linear_below_two(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch._last_settlement_outcome = "LOSS"
    orch.risk_manager.consecutive_losses_linear = 1
    orch._run_trading_cycle_if_ready = AsyncMock(return_value=True)
    recorded: list[float] = []

    async def record_sleep(seconds: float) -> None:
        recorded.append(seconds)

    with (
        patch(f"{COOLDOWN_MODULE}.asyncio.sleep", side_effect=record_sleep),
        patch_instant_post_settlement_poll(),
    ):
        await run_post_settlement_breath_and_cycle(orch)
    assert recorded == []
    orch._run_trading_cycle_if_ready.assert_awaited()


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
    assert orch.risk_manager.pending_loss.get("RDBULL") == pytest.approx(332.28)
    assert orch.risk_manager.last_loss_stake == pytest.approx(332.28)
    assert orch.risk_manager.total_session_profit == pytest.approx(-332.28)
