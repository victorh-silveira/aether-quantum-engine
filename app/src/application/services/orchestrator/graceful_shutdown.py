"""Encerramento gracioso de infraestrutura e hooks de shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from src.application.services.deep_learning.dl_deferred_train import cancel_deferred_symbol_training
from src.application.services.orchestrator.session_target_bootstrap import clear_current_session_redis_keys
from src.application.services.orchestrator.settlement_queue_ops import cancel_settlement_queue_fast
from src.application.services.orchestrator.watchdog_service import stop_ingestion_watchdog
from src.infrastructure.factories.infra_factory import close_infra_services
from src.infrastructure.inference.meta_classifier_pool import close_meta_classifier_client
from src.infrastructure.inference.triton_grpc_client import close_triton_grpc_client


_INFRA_TASK_PREFIXES = ("httpx",)
_INFRA_TASK_MARKERS = (
    "_listen",
    "_ping_loop",
    "websocketclientprotocol",
    "connectionpool",
    "_run_worker",
    "_correlation_worker",
    "safe_callback",
)
_APP_TASK_MARKERS = (
    "_run_deferred_training",
    "_subscribe_open_contract_background",
    "_run_settlement_watch",
    "run_post_settlement_breath_and_cycle",
    "_settlement_worker_loop",
    "_release_stuck_trading_slot",
    "_profit_table_audit_loop",
    "schedule_recovery_skip_counter_increment",
    "aether-watchdog",
)


async def _close_triton_if_enabled(orch: Any) -> None:
    """Fecha pool gRPC Triton quando habilitado na configuracao."""
    infra_cfg = orch.config.get("infra") if isinstance(orch.config, dict) else {}
    if not isinstance(infra_cfg, dict):
        return
    triton_cfg = infra_cfg.get("triton") if isinstance(infra_cfg.get("triton"), dict) else {}
    if not bool(triton_cfg.get("enabled", False)):
        return
    await close_triton_grpc_client()


def _task_label(task: asyncio.Task[Any]) -> str:
    """Retorna nome legivel da task para filtragem de shutdown."""
    name = task.get_name() if hasattr(task, "get_name") else ""
    if name:
        return str(name)
    coro = task.get_coro()
    if coro is None:
        return ""
    qual = getattr(coro, "__qualname__", "") or getattr(coro, "__name__", "")
    return str(qual)


def _is_infrastructure_async_task(task: asyncio.Task[Any]) -> bool:
    """Ignora tasks de rede persistente e bibliotecas HTTP internas."""
    label = _task_label(task).lower()
    if any(label.startswith(prefix) for prefix in _INFRA_TASK_PREFIXES):
        return True
    return any(marker in label for marker in _INFRA_TASK_MARKERS)


def _is_application_async_task(task: asyncio.Task[Any]) -> bool:
    """Identifica tasks explicitas da esteira de trading do motor."""
    label = _task_label(task)
    lowered = label.lower()
    if lowered == "aether-watchdog":
        return True
    return any(marker in label for marker in _APP_TASK_MARKERS)


def _orchestrator_owned_tasks(orch: Any) -> list[asyncio.Task[Any]]:
    """Coleta tasks registradas no orquestrador sem varrer o loop inteiro."""
    owned: list[asyncio.Task[Any]] = []
    for attr in (
        "_settlement_worker_task",
        "_post_settlement_task",
        "_trading_slot_poll_task",
        "_profit_table_audit_task",
    ):
        task = getattr(orch, attr, None)
        if isinstance(task, asyncio.Task) and not task.done():
            owned.append(task)
    watchdog = getattr(orch, "_ingestion_watchdog", None)
    if watchdog is not None:
        wd_task = getattr(watchdog, "_task", None)
        if isinstance(wd_task, asyncio.Task) and not wd_task.done():
            owned.append(wd_task)
    deferred = getattr(orch, "_dl_deferred_tasks", None) or {}
    for task in deferred.values():
        if isinstance(task, asyncio.Task) and not task.done():
            owned.append(task)
    return owned


def _safe_cancel_task(task: asyncio.Task[Any]) -> bool:
    """Cancela task com protecao contra RecursionError em hierarquias aninhadas."""
    if task.done():
        return False
    try:
        task.cancel()
        return True
    except RecursionError:
        return False


async def _cancel_pending_loop_tasks(orch: Any) -> None:
    """Cancela apenas tasks da esteira de trading antes do fechamento de infraestrutura."""
    current = asyncio.current_task()
    pending_map: dict[int, asyncio.Task[Any]] = {}
    for task in _orchestrator_owned_tasks(orch):
        if task is not current and not task.done():
            pending_map[id(task)] = task
    for task in asyncio.all_tasks():
        if task is current or task.done() or id(task) in pending_map:
            continue
        if _is_infrastructure_async_task(task):
            continue
        if not _is_application_async_task(task):
            continue
        pending_map[id(task)] = task
    pending = list(pending_map.values())
    if not pending:
        return
    for task in pending:
        _safe_cancel_task(task)
    await asyncio.gather(*pending, return_exceptions=True)


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
    await _cancel_pending_loop_tasks(orch)
    await close_infrastructure_connections(orch)
