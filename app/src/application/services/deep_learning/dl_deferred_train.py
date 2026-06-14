"""Retreino DL em background para nao bloquear ciclos de execucao."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from src.application.services.deep_learning.dl_bootstrap_train import (
    _bootstrap_training_context,
    _ordered_bootstrap_symbols,
)
from src.application.services.deep_learning.dl_retrain import (
    clear_force_retrain,
    reset_bars_since_train,
    should_retrain_symbol,
)
from src.application.services.deep_learning.dl_symbol_runtime import candle_epoch
from src.application.services.deep_learning.dl_symbol_train import run_symbol_training
from src.application.services.deep_learning.dl_training_gate import runtime_in_training


logger = logging.getLogger("AETH")


def try_enqueue_next_bootstrap_training(orch) -> None:
    """Agenda o proximo simbolo de bootstrap quando o treino deferido anterior termina."""
    for symbol in _ordered_bootstrap_symbols(orch):
        dl_config, params, min_len, granularity, runtime, prices, open_, high, low, micro = _bootstrap_training_context(
            orch, symbol
        )
        if not runtime_in_training(runtime, params):
            continue
        do_train, reason = should_retrain_symbol(orch, symbol, runtime, params, candle_epoch(orch, symbol))
        if not do_train or reason != "bootstrap":
            continue
        if len(prices) < min_len:
            return
        enqueue_deferred_symbol_training(
            orch,
            symbol,
            train_fn=run_symbol_training,
            train_args=(symbol, runtime, prices, dl_config, params, candle_epoch(orch, symbol), orch),
            train_kwargs={
                "granularity": granularity,
                "open_": open_,
                "high": high,
                "low": low,
                "micro": micro,
            },
        )
        return


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
        try_enqueue_next_bootstrap_training(orch)
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
