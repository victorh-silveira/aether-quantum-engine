"""Encerramento gracioso de infraestrutura e hooks de shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from typing import Any

from src.application.services.deep_learning.dl_deferred_train import cancel_deferred_symbol_training
from src.application.services.orchestrator.session_target_bootstrap import clear_current_session_redis_keys
from src.application.services.orchestrator.settlement_queue_ops import cancel_settlement_queue_fast
from src.application.services.orchestrator.watchdog_service import stop_ingestion_watchdog
from src.infrastructure.factories.infra_factory import close_infra_services
from src.infrastructure.inference.meta_classifier_pool import close_meta_classifier_client
from src.infrastructure.inference.triton_grpc_client import close_triton_grpc_client


async def _close_triton_if_enabled(orch: Any) -> None:
    """Fecha pool gRPC Triton quando habilitado na configuracao."""
    infra_cfg = orch.config.get("infra") if isinstance(orch.config, dict) else {}
    if not isinstance(infra_cfg, dict):
        return
    triton_cfg = infra_cfg.get("triton") if isinstance(infra_cfg.get("triton"), dict) else {}
    if not bool(triton_cfg.get("enabled", False)):
        return
    await close_triton_grpc_client()


async def close_infrastructure_connections(orch: Any) -> None:
    """Aguarda fechamento de Triton, Timescale, Redis e WebSocket."""
    if getattr(orch, "_infra_shutdown_done", False):
        return
    orch._infra_shutdown_done = True
    orch.running = False
    logger = logging.getLogger("AETH")
    task = getattr(orch, "_post_settlement_task", None)
    if task is not None and not task.done():
        task.cancel()
        if isinstance(task, asyncio.Task):
            with contextlib.suppress(asyncio.CancelledError):
                await task
    cancel_deferred_symbol_training(orch)
    await stop_ingestion_watchdog(orch)
    await clear_current_session_redis_keys(orch)
    await _close_triton_if_enabled(orch)
    await close_meta_classifier_client()
    infra = getattr(orch, "infra", None)
    if infra is not None:
        await close_infra_services(infra)
    ws = getattr(orch, "ws", None)
    if ws is not None:
        try:
            await ws.close()
        except Exception as exc:
            logger.debug("STOP: WebSocket close: %s", exc)
    logger.debug("STOP: infra encerrada.")


async def graceful_shutdown(orch: Any, *, fast_path: bool = False) -> None:
    """Encerra o motor; fast_path cancela filas sem aguardar handshakes pendentes."""
    orch.running = False
    if fast_path:
        cancel_settlement_queue_fast(orch)
    task = getattr(orch, "_post_settlement_task", None)
    if task is not None and not task.done():
        task.cancel()
        if not fast_path and isinstance(task, asyncio.Task):
            with contextlib.suppress(asyncio.CancelledError):
                await task
    await close_infrastructure_connections(orch)


_original_excepthook = sys.excepthook


def _shutdown_safe_excepthook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: object,
) -> None:
    """Ignora ruido de shutdown assincrono; repassa demais excecoes ao hook original."""
    if exc_type is SystemExit:
        return
    if exc_type is GeneratorExit:
        return
    if exc_type is RuntimeError:
        msg = str(exc_value)
        if "Event loop is closed" in msg or "cannot schedule new futures" in msg:
            return
    if exc_type is AttributeError and "call_exception_handler" in str(exc_value):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_closed():
        return
    _original_excepthook(exc_type, exc_value, exc_tb)


def install_shutdown_excepthook() -> None:
    """Instala excepthook que ignora ruido de GC apos loop asyncio fechado."""
    sys.excepthook = _shutdown_safe_excepthook
