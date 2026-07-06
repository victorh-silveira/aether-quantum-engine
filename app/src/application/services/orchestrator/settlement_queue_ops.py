"""Operacoes de fila de liquidacao sem dependencia de settlement_logic."""

from __future__ import annotations

from typing import Any


def cancel_settlement_queue_fast(orch: Any) -> None:
    """Cancela worker e drena fila sem aguardar handshakes pendentes."""
    worker = getattr(orch, "_settlement_worker_task", None)
    if worker is not None and not worker.done():
        worker.cancel()
    queue = getattr(orch, "_settlement_queue", None)
    if queue is not None:
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()
    orch._settlement_worker_task = None
