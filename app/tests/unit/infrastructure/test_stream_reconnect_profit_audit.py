import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.handlers.stream_reconnect_profit_audit import (
    _profit_table_audit_loop,
    schedule_profit_table_audit,
)


@pytest.mark.asyncio
async def test_profit_table_audit_loop_stops_when_not_running():
    orch = MagicMock()
    orch.running = False
    with patch(
        "src.infrastructure.handlers.stream_reconnect_profit_audit.reconcile_after_ws_recovery",
        new_callable=AsyncMock,
    ) as reconcile:
        await _profit_table_audit_loop(orch, reason="test")
        reconcile.assert_not_awaited()


@pytest.mark.asyncio
async def test_profit_table_audit_loop_success_immediate():
    orch = MagicMock()
    orch.running = True
    orch.ws = MagicMock()
    orch.ws.is_running = True
    with (
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit.reconcile_after_ws_recovery",
            new_callable=AsyncMock,
            return_value=MagicMock(errors=[], settled_count=2),
        ) as reconcile,
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        await _profit_table_audit_loop(orch, reason="broker_unavailable")
    reconcile.assert_awaited_once()


@pytest.mark.asyncio
async def test_profit_table_audit_loop_waits_for_ws_then_succeeds():
    orch = MagicMock()
    orch.running = True
    orch.ws = MagicMock()
    running = {"ok": False}
    type(orch.ws).is_running = property(lambda _self: running["ok"])

    async def _sleep(*_args, **_kwargs):
        running["ok"] = True

    with (
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit.reconcile_after_ws_recovery",
            new_callable=AsyncMock,
            return_value=MagicMock(errors=[], settled_count=2),
        ) as reconcile,
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit.asyncio.sleep",
            new_callable=AsyncMock,
            side_effect=_sleep,
        ),
    ):
        await _profit_table_audit_loop(orch, reason="broker_unavailable")
    reconcile.assert_awaited_once()


@pytest.mark.asyncio
async def test_profit_table_audit_loop_retries_on_errors():
    orch = MagicMock()
    orch.running = True
    orch.ws = MagicMock()
    orch.ws.is_running = True
    with (
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit.reconcile_after_ws_recovery",
            new_callable=AsyncMock,
            return_value=MagicMock(errors=["x"], settled_count=0),
        ) as reconcile,
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit.asyncio.sleep",
            new_callable=AsyncMock,
        ),
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit._MAX_ATTEMPTS",
            2,
        ),
    ):
        await _profit_table_audit_loop(orch, reason="stream_reconnect")
    assert reconcile.await_count == 2


@pytest.mark.asyncio
async def test_profit_table_audit_loop_handles_exception():
    orch = MagicMock()
    orch.running = True
    orch.ws = MagicMock()
    orch.ws.is_running = True
    with (
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit.reconcile_after_ws_recovery",
            new_callable=AsyncMock,
            side_effect=RuntimeError("broker indisponivel"),
        ),
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit.asyncio.sleep",
            new_callable=AsyncMock,
        ),
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit._MAX_ATTEMPTS",
            1,
        ),
    ):
        await _profit_table_audit_loop(orch, reason="broker_unavailable")


def test_schedule_profit_table_audit_skips_duplicate_task():
    orch = MagicMock()
    existing = MagicMock()
    existing.done.return_value = False
    orch._profit_table_audit_task = existing
    schedule_profit_table_audit(orch, reason="dup")
    assert orch._profit_table_audit_task is existing


@pytest.mark.asyncio
async def test_schedule_profit_table_audit_skips_active_asyncio_task():
    orch = MagicMock()
    orch.running = True

    async def _hold():
        await asyncio.Event().wait()

    task = asyncio.create_task(_hold())
    orch._profit_table_audit_task = task
    schedule_profit_table_audit(orch, reason="dup")
    assert orch._profit_table_audit_task is task
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_schedule_profit_table_audit_without_running_loop():
    schedule_profit_table_audit(MagicMock(), reason="no_loop")


@pytest.mark.asyncio
async def test_schedule_profit_table_audit_creates_task():
    orch = MagicMock()
    orch.running = True
    orch.ws = MagicMock()
    orch.ws.is_running = True
    with (
        patch(
            "src.infrastructure.handlers.stream_reconnect_profit_audit._profit_table_audit_loop",
            new_callable=AsyncMock,
        ),
        patch.object(orch, "_profit_table_audit_task", None, create=True),
    ):
        schedule_profit_table_audit(orch, reason="created")
        task = orch._profit_table_audit_task
        assert task is not None
        await task
