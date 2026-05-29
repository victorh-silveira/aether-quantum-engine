"""Agendamento de ciclo apos liquidacao com folego configuravel."""

from __future__ import annotations

import asyncio
from typing import Any


def schedule_trading_cycle_after_settlement(orch: Any) -> None:
    """Agenda novo ciclo de decisao logo apos liquidacao do contrato."""
    if not orch.running:
        return
    if orch.state.active_contracts:
        return
    if orch.is_trading:
        return
    task = orch._post_settlement_task
    if task is not None and not task.done():
        return
    orch._post_settlement_task = asyncio.create_task(run_post_settlement_breath_and_cycle(orch))


async def run_post_settlement_breath_and_cycle(orch: Any) -> None:
    """Aplica folego pos-liquidacao antes de um novo ciclo."""
    breath = float(orch.config.get("orchestrator", {}).get("post_settlement_breath_seconds", 60))
    breath = max(0.0, breath)
    if breath > 0:
        await asyncio.sleep(breath)
    if not orch.running:
        return
    if orch.state.active_contracts:
        return
    if orch.is_trading:
        return
    orch._last_cluster_cycle_end = 0.0
    await orch._run_trading_cycle_if_ready()
