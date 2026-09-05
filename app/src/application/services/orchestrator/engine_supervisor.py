"""Registro e supervisao de tasks asyncio do motor (sem fire-and-forget orfao)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any


class EngineTaskRegistry:
    """Rastreia tasks de curta/media vida para cancelamento no shutdown."""

    __slots__ = ("_tasks",)

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn[T](
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: str | None = None,
    ) -> asyncio.Task[T]:
        """Cria task nomeada e registra para cancelamento estruturado."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def cancel_all(self) -> None:
        """Cancela tasks ainda ativas e aguarda termino."""
        pending = [t for t in self._tasks if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()


def registry_of(orch: Any) -> EngineTaskRegistry:
    """Obtem ou cria o registro de tasks no orquestrador."""
    reg = getattr(orch, "_engine_task_registry", None)
    if isinstance(reg, EngineTaskRegistry):
        return reg
    reg = EngineTaskRegistry()
    orch._engine_task_registry = reg
    return reg


def spawn_background[T](
    orch: Any,
    coro: Coroutine[Any, Any, T],
    *,
    name: str | None = None,
) -> asyncio.Task[T]:
    """Atalho: spawna task rastreada no orquestrador."""
    return registry_of(orch).spawn(coro, name=name)


async def cancel_background_tasks(orch: Any) -> None:
    """Cancela todas as tasks rastreadas do orquestrador."""
    reg = getattr(orch, "_engine_task_registry", None)
    if isinstance(reg, EngineTaskRegistry):
        await reg.cancel_all()


async def shield_critical[T](awaitable: Awaitable[T]) -> T:
    """Protege secao critica (Redis/broker) contra cancelamento externo."""
    return await asyncio.shield(awaitable)


async def run_task_group(
    *workers: Callable[[], Coroutine[Any, Any, None]],
) -> None:
    """Executa workers de longa vida sob asyncio.TaskGroup (ExceptionGroup)."""
    async with asyncio.TaskGroup() as tg:
        for factory in workers:
            tg.create_task(factory())


async def cancel_task_quiet(task: asyncio.Task[Any] | None) -> None:
    """Cancela uma task isolada sem propagar CancelledError ao caller."""
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
