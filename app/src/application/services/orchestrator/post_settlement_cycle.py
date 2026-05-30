"""Agendamento de ciclo apos liquidacao com folego configuravel."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.llm.synthetic_universe import resolve_post_settlement_breath_seconds


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
    orch_cfg = orch.config.get("orchestrator") if isinstance(getattr(orch, "config", None), dict) else {}
    full_cfg = orch.config if isinstance(getattr(orch, "config", None), dict) else {}
    breath = resolve_post_settlement_breath_seconds(orch_cfg, full_cfg)
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
