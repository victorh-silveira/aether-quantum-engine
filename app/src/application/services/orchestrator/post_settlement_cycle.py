"""Agendamento de ciclo após liquidação com fôlego configurável."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.application.services.orchestrator.settlement_utils import (
    clear_contract_metadata,
    prune_orphan_contract_ids,
)


def _release_post_settlement_task(orch: Any, task: asyncio.Task) -> None:
    """Remove referencia da task pos-liquidacao quando ela encerra."""
    if orch._post_settlement_task is task:
        orch._post_settlement_task = None


def post_settlement_cycle_pending(orch: Any) -> bool:
    """True enquanto a task pos-liquidacao ainda esta em execucao."""
    task = getattr(orch, "_post_settlement_task", None)
    return task is not None and not task.done()


async def _poll_delay(seconds: float) -> None:
    """Aguarda intervalo de polling entre tentativas pos-liquidacao."""
    await asyncio.sleep(seconds)


async def _await_post_settlement_breath(orch: Any, breath: float, poll: float) -> None:
    """Aguarda folego pos-liquidacao, interrompido por reagendamento."""
    if breath <= 0:
        return
    wake = orch._post_settlement_wake
    remaining = breath
    while remaining > 0 and orch.running:
        wake.clear()
        wait = min(poll, remaining)
        try:
            await asyncio.wait_for(wake.wait(), timeout=wait)
            return
        except TimeoutError:
            remaining -= wait


def _prune_stale_risk_contract_ids(orch: Any) -> None:
    """Remove IDs de contrato orfaos do risk quando o estado local ja esta vazio."""
    if not orch.state.active_contracts and orch.risk_manager.active_contract_ids:
        kept, orphans = prune_orphan_contract_ids(
            list(orch.risk_manager.active_contract_ids),
            orch.state.active_contracts,
        )
        orch.risk_manager.active_contract_ids = kept
        if orphans:
            clear_contract_metadata(orphans, orch.risk_manager)


def schedule_trading_cycle_after_settlement(orch: Any) -> None:
    """Agenda novo ciclo de decisão logo após liquidação do contrato."""
    if not orch.running:
        return
    if orch.state.active_contracts:
        return
    _prune_stale_risk_contract_ids(orch)
    task = orch._post_settlement_task
    if task is not None and not task.done():
        orch._post_settlement_wake.set()
        return
    new_task = asyncio.create_task(run_post_settlement_breath_and_cycle(orch))
    orch._post_settlement_task = new_task
    new_task.add_done_callback(lambda done: _release_post_settlement_task(orch, done))


async def _attempt_post_settlement_trading_cycle(orch: Any, orch_cfg: dict) -> bool:
    """Executa um ciclo pos-liquidacao; True quando concluido."""
    orch._last_cluster_cycle_end = 0.0
    orch._dl_fast_cycle = True
    cycle_timeout = float(orch_cfg.get("post_settlement_cycle_timeout_seconds", 90.0))
    try:
        try:
            return await asyncio.wait_for(
                orch._run_trading_cycle_if_ready(),
                timeout=cycle_timeout,
            )
        except TimeoutError:
            orch.is_trading = False
            orch.logger.warning("CICLO: timeout pos-liquidacao (%.0fs); nova tentativa", cycle_timeout)
            return False
    finally:
        orch._dl_fast_cycle = False


async def _run_post_settlement_retry_loop(orch: Any, orch_cfg: dict, poll: float) -> None:
    """Repete tentativas de ciclo ate sucesso ou motor parar."""
    trading_wait_limit = float(orch_cfg.get("post_settlement_is_trading_wait_seconds", 120.0))
    retry_limit = float(orch_cfg.get("post_settlement_cycle_retry_seconds", 120.0))
    trading_deadline = time.monotonic() + trading_wait_limit
    retry_deadline = time.monotonic() + retry_limit
    trading_wait_logged = False
    while orch.running:
        if orch.state.active_contracts:
            trading_wait_logged = False
            await _poll_delay(poll)
            continue
        if orch.is_trading:
            if not trading_wait_logged:
                orch.logger.info("CICLO: aguardando slot pos-liquidacao")
                trading_wait_logged = True
            if time.monotonic() >= trading_deadline:
                orch.is_trading = False
                trading_deadline = time.monotonic() + trading_wait_limit
                trading_wait_logged = False
                orch.logger.warning("CICLO: is_trading preso; liberando e retentando")
            await _poll_delay(poll)
            continue
        trading_wait_logged = False
        if await _attempt_post_settlement_trading_cycle(orch, orch_cfg):
            return
        if time.monotonic() >= retry_deadline:
            retry_deadline = time.monotonic() + retry_limit
            orch.logger.warning("CICLO: ciclo pos-liquidacao incompleto; nova tentativa")
        await _poll_delay(poll)


async def run_post_settlement_breath_and_cycle(orch: Any) -> None:
    """Aplica fôlego pós-liquidação antes de um novo ciclo."""
    try:
        orch_cfg = orch.config.get("orchestrator") if isinstance(getattr(orch, "config", None), dict) else {}
        poll = 0.25
        breath = float(orch_cfg.get("post_settlement_breath_seconds", 8.0))
        await _await_post_settlement_breath(orch, breath, poll)
        await _run_post_settlement_retry_loop(orch, orch_cfg, poll)
    finally:
        current = asyncio.current_task()
        if current is not None:
            _release_post_settlement_task(orch, current)
