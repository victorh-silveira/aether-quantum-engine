import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.post_settlement_cycle import (
    _run_post_settlement_retry_loop,
    run_post_settlement_breath_and_cycle,
)
from src.application.services.orchestrator.post_settlement_loss_cooldown import (
    POST_LOSS_COOLDOWN_BASE_SECONDS,
    POST_LOSS_COOLDOWN_GROWTH,
    await_post_loss_cooldown,
    log_trading_cycle_cooldown_skip,
    orchestrator_cooldown_active,
    orchestrator_cooldown_remaining,
    post_loss_cooldown_active,
    post_loss_cooldown_blocks_trading_cycle,
    post_loss_cooldown_delay_seconds,
    schedule_post_loss_cooldown,
)
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from tests.unit.application.post_settlement_helpers import patch_instant_post_settlement_poll


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"


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
async def test_schedule_post_loss_cooldown_sets_cooldown_until(orch_ready):
    orch = orch_ready
    orch._last_settlement_outcome = "LOSS"
    orch.risk_manager.consecutive_losses_linear = 3
    expected = post_loss_cooldown_delay_seconds(3)
    loop = asyncio.get_running_loop()
    base = loop.time()
    delay = schedule_post_loss_cooldown(orch)
    assert delay == pytest.approx(expected)
    assert orch._cooldown_until - base == pytest.approx(expected, abs=0.05)
    assert orchestrator_cooldown_active(orch, now=base + 1.0) is True
    assert orchestrator_cooldown_active(orch, now=base + expected + 0.01) is False
    assert orchestrator_cooldown_remaining(orch, now=base + 1.0) == pytest.approx(expected - 1.0, abs=0.001)
    assert expected == pytest.approx(36.905625)


@pytest.mark.asyncio
async def test_await_post_loss_cooldown_skips_when_orchestrator_stopped(orch_ready):
    orch = orch_ready
    orch.running = False
    orch._last_settlement_outcome = "LOSS"
    orch.risk_manager.consecutive_losses_linear = 3
    delay = await await_post_loss_cooldown(orch)
    assert delay == 0.0
    assert getattr(orch, "_cooldown_until", 0.0) == 0.0


@pytest.mark.asyncio
async def test_await_post_loss_cooldown_skips_on_win(orch_ready):
    orch = orch_ready
    orch._last_settlement_outcome = "WIN"
    orch.risk_manager.consecutive_losses_linear = 4
    delay = await await_post_loss_cooldown(orch)
    assert delay == 0.0
    assert getattr(orch, "_cooldown_until", 0.0) == 0.0


@pytest.mark.asyncio
async def test_sequential_loss_levels_expand_post_settlement_cooldown(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    deadlines_by_linear: dict[int, float] = {}
    loop = asyncio.get_running_loop()

    for linear in (2, 3, 4):
        orch._last_settlement_outcome = "LOSS"
        orch.risk_manager.consecutive_losses_linear = linear
        base = loop.time()
        with patch_instant_post_settlement_poll():
            await run_post_settlement_breath_and_cycle(orch)
        expected = post_loss_cooldown_delay_seconds(linear)
        deadlines_by_linear[linear] = float(orch._cooldown_until)
        assert deadlines_by_linear[linear] - base == pytest.approx(expected, abs=0.05)

    assert deadlines_by_linear[3] > deadlines_by_linear[2]
    assert deadlines_by_linear[4] > deadlines_by_linear[3]


@pytest.mark.asyncio
async def test_run_post_settlement_skips_cooldown_when_linear_below_two(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch._last_settlement_outcome = "LOSS"
    orch.risk_manager.consecutive_losses_linear = 1
    orch._run_trading_cycle_if_ready = AsyncMock(return_value=True)
    with patch_instant_post_settlement_poll():
        await run_post_settlement_breath_and_cycle(orch)
    assert getattr(orch, "_cooldown_until", 0.0) == 0.0
    orch._run_trading_cycle_if_ready.assert_awaited()


@pytest.mark.asyncio
async def test_trading_cycle_skips_while_post_loss_cooldown_active(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    loop = asyncio.get_running_loop()
    orch._cooldown_until = loop.time() + 60.0
    with patch(
        f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
        new_callable=AsyncMock,
    ) as collect_mock:
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is False
    collect_mock.assert_not_awaited()
    assert post_loss_cooldown_blocks_trading_cycle(orch) is True


@pytest.mark.asyncio
async def test_post_settlement_retry_loop_defers_while_cooldown_active(orch_ready):
    orch = orch_ready
    orch._last_settlement_outcome = "LOSS"
    orch.risk_manager.consecutive_losses_linear = 3
    schedule_post_loss_cooldown(orch)
    assert orchestrator_cooldown_active(orch) is True
    cycle_mock = AsyncMock(return_value=True)
    orch_cfg = orch.config.setdefault("orchestrator", {})
    with patch.object(orch, "_run_trading_cycle_if_ready", cycle_mock):
        await _run_post_settlement_retry_loop(orch, orch_cfg, 0.25)
    cycle_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_post_settlement_schedules_cooldown_without_blocking_cycle(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch._last_settlement_outcome = "LOSS"
    orch.risk_manager.consecutive_losses_linear = 3
    expected = post_loss_cooldown_delay_seconds(3)
    loop = asyncio.get_running_loop()
    base = loop.time()
    cycle_mock = AsyncMock(return_value=True)
    with (
        patch.object(orch, "_run_trading_cycle_if_ready", cycle_mock),
        patch_instant_post_settlement_poll(),
    ):
        await run_post_settlement_breath_and_cycle(orch)
    assert orch._cooldown_until - base == pytest.approx(expected, abs=0.05)
    cycle_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_log_trading_cycle_cooldown_skip_deduplicates(orch_ready, caplog):
    orch = orch_ready
    loop = asyncio.get_running_loop()
    orch._cooldown_until = loop.time() + 30.0
    with caplog.at_level("INFO"):
        log_trading_cycle_cooldown_skip(orch)
        log_trading_cycle_cooldown_skip(orch)
    skip_logs = [record for record in caplog.records if "resfriamento pos-LOSS ativo" in record.message]
    assert len(skip_logs) == 1


def test_orchestrator_cooldown_remaining_zero_when_inactive():
    orch = type("Orch", (), {})()
    assert orchestrator_cooldown_remaining(orch) == 0.0


def test_log_trading_cycle_cooldown_skip_noop_without_deadline(orch_ready, caplog):
    orch = orch_ready
    with caplog.at_level("INFO"):
        log_trading_cycle_cooldown_skip(orch)
    assert not [record for record in caplog.records if "resfriamento pos-LOSS ativo" in record.message]
