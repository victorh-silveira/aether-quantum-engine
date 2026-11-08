import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.graceful_shutdown import (
    _cancel_pending_loop_tasks,
    _is_application_async_task,
    _is_infrastructure_async_task,
    _orchestrator_owned_tasks,
    _safe_cancel_task,
    _task_label,
)


def _empty_orch() -> MagicMock:
    orch = MagicMock()
    orch._settlement_worker_task = None
    orch._post_settlement_task = None
    orch._trading_slot_poll_task = None
    orch._profit_table_audit_task = None
    orch._ingestion_watchdog = None
    orch._dl_deferred_tasks = {}
    return orch


def test_task_label_uses_name_and_coro_fallbacks():
    named = MagicMock()
    named.get_name.return_value = "aether-watchdog"
    assert _task_label(named) == "aether-watchdog"
    unnamed = MagicMock()
    unnamed.get_name.return_value = ""
    unnamed.get_coro.return_value = None
    assert _task_label(unnamed) == ""
    coro_task = MagicMock()
    coro_task.get_name.return_value = ""
    coro_task.get_coro.return_value = SimpleNamespace(__qualname__="run_post_settlement_breath_and_cycle")
    assert _task_label(coro_task) == "run_post_settlement_breath_and_cycle"


def test_infrastructure_and_application_task_filters():
    infra = MagicMock()
    infra.get_name.return_value = "httpx_client"
    assert _is_infrastructure_async_task(infra) is True
    app = MagicMock()
    app.get_name.return_value = "_run_settlement_watch"
    assert _is_infrastructure_async_task(app) is False
    assert _is_application_async_task(app) is True
    watchdog = MagicMock()
    watchdog.get_name.return_value = "aether-watchdog"
    assert _is_application_async_task(watchdog) is True


def test_safe_cancel_task_handles_done_and_recursion():
    done_task = MagicMock()
    done_task.done.return_value = True
    assert _safe_cancel_task(done_task) is False
    rec_task = MagicMock()
    rec_task.done.return_value = False
    rec_task.cancel.side_effect = RecursionError()
    assert _safe_cancel_task(rec_task) is False


@pytest.mark.asyncio
async def test_orchestrator_owned_tasks_collects_registered_tree():
    orch = _empty_orch()

    async def _worker():
        await asyncio.sleep(3600)

    worker = asyncio.create_task(_worker(), name="_settlement_worker_loop")
    deferred = asyncio.create_task(_worker(), name="_run_deferred_training")
    wd = asyncio.create_task(_worker(), name="aether-watchdog")
    orch._settlement_worker_task = worker
    orch._ingestion_watchdog = SimpleNamespace(_task=wd)
    orch._dl_deferred_tasks = {"R_10": deferred}
    owned = _orchestrator_owned_tasks(orch)
    assert worker in owned
    assert deferred in owned
    assert wd in owned
    for task in (worker, deferred, wd):
        task.cancel()
    await asyncio.gather(worker, deferred, wd, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_pending_loop_tasks_noop_when_empty():
    await _cancel_pending_loop_tasks(_empty_orch())


@pytest.mark.asyncio
async def test_cancel_pending_loop_tasks_cancels_pending_tasks():
    mock_task = MagicMock()
    mock_task.done.return_value = False
    mock_task.get_name.return_value = "_settlement_worker_loop"
    current = asyncio.current_task()
    with (
        patch("asyncio.all_tasks", return_value={mock_task, current}),
        patch("asyncio.current_task", return_value=current),
        patch("asyncio.gather", new_callable=AsyncMock, return_value=[]),
    ):
        await _cancel_pending_loop_tasks(_empty_orch())
    mock_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_pending_loop_tasks_includes_owned_tasks_before_scan():
    owned_task = MagicMock()
    owned_task.done.return_value = False
    owned_task.get_name.return_value = "_settlement_worker_loop"
    current = asyncio.current_task()
    with (
        patch(
            "src.application.services.orchestrator.graceful_shutdown._orchestrator_owned_tasks",
            return_value=[owned_task],
        ),
        patch("asyncio.all_tasks", return_value={current}),
        patch("asyncio.current_task", return_value=current),
        patch("asyncio.gather", new_callable=AsyncMock, return_value=[]),
    ):
        await _cancel_pending_loop_tasks(_empty_orch())
    owned_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_pending_loop_tasks_ignores_httpx_tasks():
    infra_task = MagicMock()
    infra_task.done.return_value = False
    infra_task.get_name.return_value = "httpx_pool"
    app_task = MagicMock()
    app_task.done.return_value = False
    app_task.get_name.return_value = "_run_settlement_watch"
    current = asyncio.current_task()
    with (
        patch("asyncio.all_tasks", return_value={infra_task, app_task, current}),
        patch("asyncio.current_task", return_value=current),
        patch("asyncio.gather", new_callable=AsyncMock, return_value=[]),
    ):
        await _cancel_pending_loop_tasks(_empty_orch())
    infra_task.cancel.assert_not_called()
    app_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_pending_loop_tasks_filters_taskgroup_hierarchy():
    orch = _empty_orch()
    started = asyncio.Event()

    async def _app_parent() -> None:
        started.set()
        try:
            async with asyncio.TaskGroup() as tg:

                async def _child() -> None:
                    await asyncio.sleep(3600)

                tg.create_task(_child())
        except asyncio.CancelledError:
            raise

    parent = asyncio.create_task(_app_parent(), name="_settlement_worker_loop")
    await started.wait()
    await _cancel_pending_loop_tasks(orch)
    assert parent.cancelled() or parent.done()
