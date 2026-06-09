"""Retreino DL em background para nao bloquear ciclos de execucao."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from src.application.services.deep_learning.dl_retrain import clear_force_retrain, reset_bars_since_train


logger = logging.getLogger("AETH")


def _deferred_tasks(orch) -> dict[str, asyncio.Task]:
    """Retorna o mapa de tasks de retreino deferido do orquestrador."""
    bag = getattr(orch, "_dl_deferred_tasks", None)
    if bag is None:
        orch._dl_deferred_tasks = {}
        bag = orch._dl_deferred_tasks
    return bag


def _training_semaphore(orch) -> asyncio.Semaphore:
    """Retorna semaforo que serializa retreinos deferidos do orquestrador."""
    sem = getattr(orch, "_dl_deferred_sem", None)
    if sem is None:
        orch._dl_deferred_sem = asyncio.Semaphore(1)
        sem = orch._dl_deferred_sem
    return sem


async def _run_deferred_training(
    orch,
    symbol: str,
    train_fn: Callable[..., Any],
    train_args: tuple,
    train_kwargs: dict,
) -> None:
    """Executa retreino em thread e limpa flags de reagendamento do simbolo."""
    try:
        async with _training_semaphore(orch):
            await asyncio.to_thread(train_fn, *train_args, **train_kwargs)
        clear_force_retrain(orch, symbol)
        reset_bars_since_train(orch, symbol)
    except Exception as exc:
        logger.error("DL: retreino deferido %s falhou: %s", symbol, exc)


def enqueue_deferred_symbol_training(
    orch,
    symbol: str,
    *,
    train_fn: Callable[..., Any],
    train_args: tuple,
    train_kwargs: dict,
) -> None:
    """Agenda retreino pesado em background para nao bloquear o ciclo de execucao."""
    sym = str(symbol)
    tasks = _deferred_tasks(orch)
    existing = tasks.get(sym)
    if existing is not None and not existing.done():
        return
    for pending in tasks.values():
        if pending is not None and not pending.done():
            return
    tasks[sym] = asyncio.create_task(_run_deferred_training(orch, sym, train_fn, train_args, train_kwargs))


def cancel_deferred_symbol_training(orch) -> None:
    """Cancela tasks de retreino deferido pendentes."""
    tasks = getattr(orch, "_dl_deferred_tasks", None)
    if not tasks:
        return
    for task in list(tasks.values()):
        if not task.done():
            task.cancel()
    tasks.clear()
