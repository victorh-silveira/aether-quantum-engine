"""Fila assincrona de liquidacoes para nao bloquear o loop principal."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.orchestrator.settlement_logic import process_contract_settlement


async def start_settlement_worker(orch: Any) -> None:
    """Inicia worker que consome liquidacoes de contrato em background."""
    task = getattr(orch, "_settlement_worker_task", None)
    if task is not None and not task.done():
        return
    orch._settlement_queue = asyncio.Queue()
    orch._settlement_worker_task = asyncio.create_task(
        _settlement_worker_loop(orch),
        name="aether-settlement-worker",
    )


async def enqueue_contract_settlement(orch: Any, data: dict) -> None:
    """Enfileira payload de liquidacao para processamento fora do hot path."""
    queue = getattr(orch, "_settlement_queue", None)
    if queue is None:
        await process_contract_settlement(orch, data)
        return
    await queue.put(data)


async def _settlement_worker_loop(orch: Any) -> None:
    """Consome fila de settlements ate o motor encerrar."""
    queue: asyncio.Queue = orch._settlement_queue
    while orch.running or not queue.empty():
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=0.25)
        except TimeoutError:
            continue
        except asyncio.CancelledError:
            raise
        try:
            await process_contract_settlement(orch, payload)
        finally:
            queue.task_done()
