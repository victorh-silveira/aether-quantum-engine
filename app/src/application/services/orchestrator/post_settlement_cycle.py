"""Agendamento de ciclo após liquidação com fôlego configurável."""

from __future__ import annotations

import asyncio
import time
from typing import Any


def _release_post_settlement_task(orch: Any, task: asyncio.Task) -> None:
    """Remove referencia da task pos-liquidacao quando ela encerra."""
    if orch._post_settlement_task is task:
        orch._post_settlement_task = None


def schedule_trading_cycle_after_settlement(orch: Any) -> None:
    """Agenda novo ciclo de decisão logo após liquidação do contrato."""
    if not orch.running:
        return
    if orch.state.active_contracts:
        return
    task = orch._post_settlement_task
    if task is not None and not task.done():
        return
    new_task = asyncio.create_task(run_post_settlement_breath_and_cycle(orch))
    orch._post_settlement_task = new_task
    new_task.add_done_callback(lambda done: _release_post_settlement_task(orch, done))


async def run_post_settlement_breath_and_cycle(orch: Any) -> None:
    """Aplica fôlego pós-liquidação antes de um novo ciclo."""
    try:
        orch_cfg = orch.config.get("orchestrator") if isinstance(getattr(orch, "config", None), dict) else {}
        breath = float(orch_cfg.get("post_settlement_breath_seconds", 8.0))
        if breath > 0:
            await asyncio.sleep(breath)
        if not orch.running:
            return
        if orch.state.active_contracts:
            return
        wait_limit = float(orch_cfg.get("post_settlement_is_trading_wait_seconds", 120.0))
        deadline = time.monotonic() + wait_limit
        while orch.running and not orch.state.active_contracts:
            if not orch.is_trading:
                orch._last_cluster_cycle_end = 0.0
                await orch._run_trading_cycle_if_ready()
                return
            if time.monotonic() >= deadline:
                schedule_trading_cycle_after_settlement(orch)
                return
            await asyncio.sleep(0.25)
    finally:
        current = asyncio.current_task()
        if current is not None:
            _release_post_settlement_task(orch, current)
