from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.post_settlement_cycle import (
    run_post_settlement_breath_and_cycle,
    schedule_trading_cycle_after_settlement,
)
from tests.unit.application.post_settlement_helpers import (
    TRADING_CYCLE_COLLECT,
    _yield_to_event_loop,
    patch_incrementing_monotonic,
    patch_instant_post_settlement_poll,
    patch_post_settlement_poll_stop_after,
)


POST_SETTLEMENT_MODULE = "src.application.services.orchestrator.post_settlement_cycle"


@pytest.mark.asyncio
async def test_run_post_settlement_timeout_releases_is_trading_and_retries(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.config["orchestrator"]["post_settlement_cycle_timeout_seconds"] = 0.01
    poll_calls = 0

    async def stop_after_poll(*_args, **_kwargs):
        nonlocal poll_calls
        poll_calls += 1
        if poll_calls >= 2:
            orch.running = False
        await _yield_to_event_loop()

    with (
        patch.object(orch, "_run_trading_cycle_if_ready", new_callable=AsyncMock, return_value=False),
        patch(
            f"{POST_SETTLEMENT_MODULE}.asyncio.wait_for",
            side_effect=TimeoutError,
        ),
        patch(f"{POST_SETTLEMENT_MODULE}._poll_delay", side_effect=stop_after_poll),
    ):
        await run_post_settlement_breath_and_cycle(orch)

    assert orch.is_trading is False


@pytest.mark.asyncio
async def test_run_post_settlement_retries_when_cycle_does_not_complete(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.config["orchestrator"]["settlement_tolerance_window_seconds"] = 1.0
    attempts = 0

    async def cycle_side_effect():
        nonlocal attempts
        attempts += 1
        if attempts >= 2:
            orch.running = False
        return False

    with (
        patch.object(orch, "_run_trading_cycle_if_ready", side_effect=cycle_side_effect),
        patch(
            f"{POST_SETTLEMENT_MODULE}.SettlementOrphanCleaner.reconcile_stale_contracts",
            new_callable=AsyncMock,
            return_value=0,
        ),
        patch(f"{POST_SETTLEMENT_MODULE}._await_post_settlement_breath", new_callable=AsyncMock),
        patch_incrementing_monotonic(step=0.6),
        patch_instant_post_settlement_poll(),
    ):
        await run_post_settlement_breath_and_cycle(orch)

    assert attempts >= 2
    assert orch._post_settlement_deadlock is False


@pytest.mark.asyncio
async def test_post_settlement_uses_ensure_future_for_trading_slot(orch_ready):
    orch = orch_ready
    orch.is_trading = True
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.config["orchestrator"]["post_settlement_is_trading_wait_seconds"] = 0.01
    polls = 0

    async def release_on_poll(*_args, **_kwargs):
        nonlocal polls
        polls += 1
        if polls >= 1:
            orch.is_trading = False
            orch.running = False
        await _yield_to_event_loop()

    with (
        patch(f"{POST_SETTLEMENT_MODULE}.asyncio.ensure_future") as ensure_future,
        patch(f"{POST_SETTLEMENT_MODULE}._poll_delay", side_effect=release_on_poll),
    ):
        ensure_future.return_value = MagicMock(done=MagicMock(return_value=True))
        await run_post_settlement_breath_and_cycle(orch)
    ensure_future.assert_called()


@pytest.mark.asyncio
async def test_run_post_settlement_releases_stuck_is_trading(orch_ready):
    orch = orch_ready
    orch.is_trading = True
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.config["orchestrator"]["post_settlement_is_trading_wait_seconds"] = 0.01

    with (
        patch_incrementing_monotonic(),
        patch_post_settlement_poll_stop_after(orch, 2),
    ):
        await run_post_settlement_breath_and_cycle(orch)

    assert orch.is_trading is False


@pytest.mark.asyncio
async def test_run_post_settlement_enables_fast_dl_cycle(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    seen_fast: list[bool] = []

    async def capture_fast_cycle():
        seen_fast.append(bool(getattr(orch, "_dl_fast_cycle", False)))
        orch._last_cycle_cluster_executed = True
        return True

    with (
        patch.object(orch, "_run_trading_cycle_if_ready", side_effect=capture_fast_cycle),
        patch_instant_post_settlement_poll(),
    ):
        await run_post_settlement_breath_and_cycle(orch)

    assert seen_fast == [True]
    assert orch._dl_fast_cycle is False


@pytest.mark.asyncio
async def test_run_post_settlement_retries_until_cycle_runs(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["post_settlement_breath_seconds"] = 0
    orch.config["orchestrator"]["post_settlement_cycle_retry_seconds"] = 2.0
    allows = iter([False, True])

    with (
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.trading_cycle_entry_allowed",
            side_effect=lambda _orch: next(allows),
        ),
        patch(
            TRADING_CYCLE_COLLECT,
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_instant_post_settlement_poll(),
    ):
        orch.executor.execute_cluster = AsyncMock()
        await run_post_settlement_breath_and_cycle(orch)

    orch.executor.execute_cluster.assert_awaited_once()


@pytest.mark.asyncio
async def test_schedule_prunes_stale_risk_ids(orch_ready):
    orch = orch_ready
    orch.risk_manager.active_contract_ids = [999]
    orch.risk_manager.contract_to_symbol[999] = "OTC_SPC"
    with (
        patch(
            TRADING_CYCLE_COLLECT,
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_instant_post_settlement_poll(),
    ):
        orch.executor.execute_cluster = AsyncMock()
        schedule_trading_cycle_after_settlement(orch)
        await orch._post_settlement_task
    assert orch.risk_manager.active_contract_ids == []
    orch.executor.execute_cluster.assert_awaited_once()


@pytest.mark.asyncio
async def test_schedule_spawns_task_even_when_is_trading(orch_ready):
    orch = orch_ready
    orch.is_trading = True
    orch.config["orchestrator"]["post_settlement_is_trading_wait_seconds"] = 0.01
    with (
        patch(
            TRADING_CYCLE_COLLECT,
            new_callable=AsyncMock,
            return_value={},
        ),
        patch_incrementing_monotonic(),
        patch_instant_post_settlement_poll(),
    ):
        orch.executor.execute_cluster = AsyncMock()
        schedule_trading_cycle_after_settlement(orch)
        assert orch._post_settlement_task is not None
        await orch._post_settlement_task
    orch.executor.execute_cluster.assert_awaited_once()
