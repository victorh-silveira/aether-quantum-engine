import asyncio
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.orchestrator_settlement_queue import (
    SettlementOrphanCleaner,
    _known_contract_ids,
    _maybe_run_orphan_cleaner,
    _settlement_worker_loop,
    enqueue_contract_settlement,
    next_settlement_backoff_seconds,
    resolve_settlement_tolerance_window,
    start_settlement_worker,
)


real_sleep = asyncio.sleep


def test_settlement_tolerance_window_and_backoff_defaults():
    assert resolve_settlement_tolerance_window(None, {}) == pytest.approx(180.0)
    assert resolve_settlement_tolerance_window(None, {"settlement_tolerance_window_seconds": 90}) == pytest.approx(90.0)
    assert next_settlement_backoff_seconds(0) == pytest.approx(1.0)
    assert next_settlement_backoff_seconds(1) == pytest.approx(2.0)
    assert next_settlement_backoff_seconds(10) == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_settlement_orphan_cleaner_reconciles_missing_portfolio_ids(orch_ready):
    orch = orch_ready
    orch.ws.is_running = True
    orch.risk_manager.active_contract_ids = [101]
    orch.risk_manager.contract_to_symbol[101] = "RDBEAR"
    orch.state.active_contracts = {101: object()}
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.fetch_portfolio",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.reconcile_single_contract",
            new_callable=AsyncMock,
            return_value=True,
        ) as reconcile_mock,
    ):
        settled = await SettlementOrphanCleaner(orch).reconcile_stale_contracts(timeout=1.0)
    assert settled == 1
    reconcile_mock.assert_awaited_once_with(orch, 101)


@pytest.mark.asyncio
async def test_settlement_orphan_cleaner_prune_guards(orch_ready):
    orch = orch_ready
    cleaner = SettlementOrphanCleaner(orch)
    orch.risk_manager = None
    cleaner._prune_local_orphans()
    orch.risk_manager = SimpleNamespace(active_contract_ids=[])
    orch.state = SimpleNamespace(active_contracts={})
    cleaner._prune_local_orphans()
    orch.risk_manager.active_contract_ids = [1]
    orch.state.active_contracts = {1: object()}
    cleaner._prune_local_orphans()
    assert orch.risk_manager.active_contract_ids == [1]


def test_known_contract_ids_skips_invalid_and_missing_risk(orch_ready):
    orch = SimpleNamespace(state=SimpleNamespace(active_contracts={"bad": 1, "42": object()}), risk_manager=None)
    assert _known_contract_ids(orch) == [42]
    orch.risk_manager = SimpleNamespace(active_contract_ids=["x", 7], contract_to_symbol={"y": "RDBEAR", 9: "RDBULL"})
    assert _known_contract_ids(orch) == [7, 9, 42]


@pytest.mark.asyncio
async def test_settlement_orphan_cleaner_handles_portfolio_and_reconcile_errors(orch_ready):
    orch = orch_ready
    orch.ws.is_running = True
    orch.risk_manager.active_contract_ids = [101, 102]
    orch.state.active_contracts = {}
    with patch(
        "src.application.services.orchestrator.orchestrator_settlement_queue.fetch_portfolio",
        new_callable=AsyncMock,
        side_effect=RuntimeError("ws down"),
    ):
        assert await SettlementOrphanCleaner(orch).reconcile_stale_contracts() == 0
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.fetch_portfolio",
            new_callable=AsyncMock,
            return_value=[{"contract_id": 101}],
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.reconcile_single_contract",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ),
    ):
        settled = await SettlementOrphanCleaner(orch).reconcile_stale_contracts()
    assert settled == 0


@pytest.mark.asyncio
async def test_settlement_passive_reconcile_clears_when_no_open_contracts(orch_ready):
    orch = orch_ready
    orch.ws.is_running = True
    orch.risk_manager.active_contract_ids = [77]
    orch.state.active_contracts = {77: object()}
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.process_redis_settlement_queue",
            new_callable=AsyncMock,
        ) as redis_mock,
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.fetch_portfolio",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.reconcile_single_contract",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue._known_contract_ids",
            side_effect=[[77], [], []],
        ),
    ):
        orch.state.active_contracts = {}
        cleared = await SettlementOrphanCleaner(orch).passive_reconcile(timeout=1.0)
    assert cleared is True
    redis_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_settlement_orphan_cleaner_skips_when_ws_missing(orch_ready):
    orch = orch_ready
    orch.ws = None
    settled = await SettlementOrphanCleaner(orch).reconcile_stale_contracts()
    assert settled == 0


@pytest.mark.asyncio
async def test_settlement_orphan_cleaner_skips_when_ws_down(orch_ready):
    orch = orch_ready
    orch.ws.is_running = False
    orch.risk_manager.active_contract_ids = [55]
    settled = await SettlementOrphanCleaner(orch).reconcile_stale_contracts()
    assert settled == 0


@pytest.mark.asyncio
async def test_settlement_orphan_cleaner_prunes_when_no_known_ids(orch_ready):
    orch = orch_ready
    orch.ws.is_running = True
    orch.risk_manager.active_contract_ids = [999]
    orch.state.active_contracts = {}
    with patch(
        "src.application.services.orchestrator.orchestrator_settlement_queue._known_contract_ids",
        side_effect=[[], []],
    ):
        settled = await SettlementOrphanCleaner(orch).reconcile_stale_contracts()
    assert settled == 0
    assert orch.risk_manager.active_contract_ids == []


@pytest.mark.asyncio
async def test_maybe_run_orphan_cleaner_resets_when_no_known(orch_ready):
    orch = orch_ready
    orch.risk_manager.active_contract_ids = []
    orch.state.active_contracts = {}
    orch._settlement_pending_since = 12.0
    await _maybe_run_orphan_cleaner(orch)
    assert orch._settlement_pending_since == 0.0


@pytest.mark.asyncio
async def test_maybe_run_orphan_cleaner_waits_inside_window(orch_ready):
    orch = orch_ready
    orch.risk_manager.active_contract_ids = [7]
    orch._settlement_pending_since = 100.0
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.time.monotonic",
            return_value=150.0,
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.SettlementOrphanCleaner.reconcile_stale_contracts",
            new_callable=AsyncMock,
        ) as cleaner,
    ):
        await _maybe_run_orphan_cleaner(orch)
    cleaner.assert_not_awaited()


async def _stop_settlement_worker(orch) -> None:
    orch.running = False
    task = getattr(orch, "_settlement_worker_task", None)
    if task is not None and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_enqueue_contract_settlement_uses_queue(orch_ready):
    orch = orch_ready
    orch.running = True
    await start_settlement_worker(orch)
    try:
        with (
            patch(
                "src.application.services.orchestrator.orchestrator_settlement_queue.process_contract_settlement",
                new_callable=AsyncMock,
            ) as settle_mock,
            patch(
                "src.application.services.orchestrator.orchestrator_settlement_queue.process_redis_settlement_queue",
                new_callable=AsyncMock,
            ),
        ):
            await enqueue_contract_settlement(orch, {"proposal_open_contract": {"contract_id": 1}})
            await asyncio.wait_for(orch._settlement_queue.join(), timeout=1.0)
        settle_mock.assert_awaited_once()
    finally:
        await _stop_settlement_worker(orch)


@pytest.mark.asyncio
async def test_enqueue_contract_settlement_fallback_without_worker(orch_ready):
    orch = orch_ready
    orch._settlement_queue = None
    with patch(
        "src.application.services.orchestrator.orchestrator_settlement_queue.process_contract_settlement",
        new_callable=AsyncMock,
    ) as settle_mock:
        await enqueue_contract_settlement(orch, {"proposal_open_contract": {"contract_id": 2}})
    settle_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_settlement_worker_is_idempotent(orch_ready):
    orch = orch_ready
    orch.running = True
    await start_settlement_worker(orch)
    try:
        first = orch._settlement_worker_task
        await start_settlement_worker(orch)
        assert orch._settlement_worker_task is first
    finally:
        await _stop_settlement_worker(orch)


@pytest.mark.asyncio
async def test_settlement_worker_continues_after_queue_timeout(orch_ready):
    orch = orch_ready
    orch.running = True
    orch._settlement_queue = asyncio.Queue()

    with (
        patch("asyncio.sleep", new=real_sleep),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue.process_redis_settlement_queue",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.orchestrator.orchestrator_settlement_queue._maybe_run_orphan_cleaner",
            new_callable=AsyncMock,
        ),
    ):
        task = asyncio.create_task(_settlement_worker_loop(orch))

        await real_sleep(0.35)

        orch.running = False
        await task
