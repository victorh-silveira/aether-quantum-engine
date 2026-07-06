import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.application.services.orchestrator.graceful_shutdown as graceful_shutdown_module
from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.graceful_shutdown import (
    _shutdown_safe_excepthook,
    close_infrastructure_connections,
    install_shutdown_excepthook,
)
from src.application.services.orchestrator.orchestrator_run_loop import stop_orchestrator
from src.infrastructure.inference.triton_grpc_client import TritonGrpcClient


@pytest.mark.asyncio
async def test_close_infrastructure_connections_idempotent():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch._post_settlement_task = None
    orch.config = {"infra": {"triton": {"enabled": True}}}
    orch.infra = MagicMock()
    orch.ws = AsyncMock()
    with (
        patch(
            "src.application.services.orchestrator.graceful_shutdown.close_triton_grpc_client",
            new_callable=AsyncMock,
        ) as close_triton,
        patch(
            "src.application.services.orchestrator.graceful_shutdown.close_infra_services",
            new_callable=AsyncMock,
        ) as close_infra,
        patch(
            "src.application.services.orchestrator.graceful_shutdown.cancel_deferred_symbol_training",
        ),
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
    close_triton.assert_awaited_once()
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
    orch.config = {"infra": {"triton": {"enabled": False}}}
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
        patch(
            "src.application.services.orchestrator.graceful_shutdown.cancel_deferred_symbol_training",
        ),
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
    orch.config = {"infra": {"triton": {"enabled": False}}}
    orch.infra = None
    orch.ws = AsyncMock()

    async def _block():
        await asyncio.sleep(60)

    task = asyncio.create_task(_block())
    orch._post_settlement_task = task
    with (
        patch(
            "src.application.services.orchestrator.graceful_shutdown.cancel_deferred_symbol_training",
        ),
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
async def test_close_infrastructure_skips_triton_when_infra_block_invalid():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch._post_settlement_task = None
    orch.config = {"infra": "invalid"}
    orch.infra = None
    orch.ws = None
    with (
        patch(
            "src.application.services.orchestrator.graceful_shutdown.close_triton_grpc_client",
            new_callable=AsyncMock,
        ) as close_triton,
        patch(
            "src.application.services.orchestrator.graceful_shutdown.cancel_deferred_symbol_training",
        ),
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
    close_triton.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_infrastructure_ws_close_failure_logged():
    orch = MagicMock()
    orch._infra_shutdown_done = False
    orch.running = True
    orch._post_settlement_task = None
    orch.config = {"infra": {"triton": {"enabled": False}}}
    orch.infra = None
    orch.ws = AsyncMock()
    orch.ws.close = AsyncMock(side_effect=RuntimeError("ws down"))
    with (
        patch(
            "src.application.services.orchestrator.graceful_shutdown.cancel_deferred_symbol_training",
        ),
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


@pytest.mark.asyncio
async def test_triton_close_channel_pool():
    with patch(
        "src.infrastructure.inference.triton_grpc_client.close_triton_grpc_client",
        new_callable=AsyncMock,
    ) as close_pool:
        await TritonGrpcClient.close_channel_pool()
    close_pool.assert_awaited_once()


@pytest.mark.asyncio
async def test_triton_close_channel_instance():
    client = TritonGrpcClient()
    with patch.object(client, "close", new_callable=AsyncMock) as close_mock:
        await client.close_channel()
    close_mock.assert_awaited_once()


def test_shutdown_safe_excepthook_ignores_closed_loop():
    gs = graceful_shutdown_module

    old_hook = sys.excepthook
    try:
        gs._original_excepthook = lambda *a: None
        install_shutdown_excepthook()
        assert sys.excepthook is _shutdown_safe_excepthook
        sys.excepthook(RuntimeError, RuntimeError("Event loop is closed"), None)
        sys.excepthook(SystemExit, SystemExit(0), None)
    finally:
        sys.excepthook = old_hook


def test_shutdown_safe_excepthook_covers_branches():
    gs = graceful_shutdown_module

    old_hook = sys.excepthook
    called: list[type[BaseException]] = []
    try:
        gs._original_excepthook = lambda exc_type, exc_value, exc_tb: called.append(exc_type)
        install_shutdown_excepthook()
        sys.excepthook(GeneratorExit, GeneratorExit(), None)
        sys.excepthook(
            AttributeError,
            AttributeError("call_exception_handler failed"),
            None,
        )
        sys.excepthook(RuntimeError, RuntimeError("cannot schedule new futures after shutdown"), None)
        sys.excepthook(ValueError, ValueError("real"), None)
        assert ValueError in called
    finally:
        sys.excepthook = old_hook


def test_shutdown_safe_excepthook_closed_loop():
    gs = graceful_shutdown_module

    old_hook = sys.excepthook
    try:
        gs._original_excepthook = lambda *a: None
        install_shutdown_excepthook()

        class _ClosedLoop:
            def is_closed(self):
                return True

        with patch("asyncio.get_running_loop", return_value=_ClosedLoop()):
            sys.excepthook(RuntimeError, RuntimeError("other"), None)
    finally:
        sys.excepthook = old_hook
