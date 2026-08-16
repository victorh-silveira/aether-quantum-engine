import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.graceful_shutdown import close_infrastructure_connections
from src.application.services.orchestrator.orchestrator_run_loop import stop_orchestrator


@pytest.mark.asyncio
async def test_close_infrastructure_connections_idempotent():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch._post_settlement_task = None
    orch.config = {"infra": {}}
    orch.infra = MagicMock()
    orch.ws = AsyncMock()
    with (
        patch(
            "src.application.services.orchestrator.graceful_shutdown.close_infra_services",
            new_callable=AsyncMock,
        ) as close_infra,
        patch("src.application.services.orchestrator.graceful_shutdown.cancel_deferred_symbol_training"),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.stop_ingestion_watchdog",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.clear_current_session_redis_keys",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.close_meta_classifier_client",
            new_callable=AsyncMock,
        ) as close_meta,
    ):
        await close_infrastructure_connections(orch)
        await close_infrastructure_connections(orch)
    close_infra.assert_awaited_once()
    close_meta.assert_awaited_once()
    orch.ws.close.assert_awaited_once()
    assert orch._infra_shutdown_done is True
    assert orch.running is False


@pytest.mark.asyncio
async def test_close_infrastructure_cancels_post_settlement_task():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch.config = {"infra": {}}
    orch.infra = MagicMock()
    orch.ws = AsyncMock()

    async def _slow():
        await asyncio.sleep(10)

    task = asyncio.create_task(_slow())
    orch._post_settlement_task = task
    with (
        patch(
            "src.application.services.orchestrator.graceful_shutdown.close_infra_services",
            new_callable=AsyncMock,
        ),
        patch("src.application.services.orchestrator.graceful_shutdown.cancel_deferred_symbol_training"),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.stop_ingestion_watchdog",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.clear_current_session_redis_keys",
            new_callable=AsyncMock,
        ),
    ):
        await close_infrastructure_connections(orch)
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_close_infrastructure_awaits_cancelled_asyncio_task():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch.config = {"infra": {}}
    orch.infra = None
    orch.ws = AsyncMock()

    async def _block():
        await asyncio.sleep(60)

    task = asyncio.create_task(_block())
    orch._post_settlement_task = task
    with (
        patch("src.application.services.orchestrator.graceful_shutdown.cancel_deferred_symbol_training"),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.stop_ingestion_watchdog",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.clear_current_session_redis_keys",
            new_callable=AsyncMock,
        ),
    ):
        await close_infrastructure_connections(orch)
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_close_infrastructure_with_invalid_infra_block():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch._post_settlement_task = None
    orch.config = {"infra": "invalid"}
    orch.infra = None
    orch.ws = None
    with (
        patch("src.application.services.orchestrator.graceful_shutdown.cancel_deferred_symbol_training"),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.stop_ingestion_watchdog",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.clear_current_session_redis_keys",
            new_callable=AsyncMock,
        ),
    ):
        await close_infrastructure_connections(orch)
    assert orch._infra_shutdown_done is True


@pytest.mark.asyncio
async def test_close_infrastructure_ws_close_failure_logged():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch._post_settlement_task = None
    orch.config = {"infra": {}}
    orch.infra = None
    orch.ws = AsyncMock()
    orch.ws.close = AsyncMock(side_effect=RuntimeError("ws down"))
    with (
        patch("src.application.services.orchestrator.graceful_shutdown.cancel_deferred_symbol_training"),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.stop_ingestion_watchdog",
            new_callable=AsyncMock,
        ),
        patch(
            "src.application.services.orchestrator.graceful_shutdown.clear_current_session_redis_keys",
            new_callable=AsyncMock,
        ),
    ):
        await close_infrastructure_connections(orch)
    orch.ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_close_infrastructure_connections(orch_config):
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
    with patch(
        "src.application.services.orchestrator.close_infrastructure_connections",
        new_callable=AsyncMock,
    ) as close_mock:
        await orch.close_infrastructure_connections()
    close_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_orchestrator_delegates_shutdown():
    orch = MagicMock()
    with patch(
        "src.application.services.orchestrator.orchestrator_run_loop.close_infrastructure_connections",
        new_callable=AsyncMock,
    ) as close_mock:
        await stop_orchestrator(orch)
    close_mock.assert_awaited_once()
