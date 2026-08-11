"""Agendamento de ciclo apos liquidacao com janela de tolerancia e backoff."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.application.services.log_dedupe import log_debug_if_changed
from src.application.services.orchestrator.graceful_shutdown import graceful_shutdown
from src.application.services.orchestrator.orchestrator_data_signature import seconds_until_next_signature_boundary
from src.application.services.orchestrator.orchestrator_settlement_queue import (
    SettlementOrphanCleaner,
    next_settlement_backoff_seconds,
    resolve_settlement_tolerance_window,
)
from src.application.services.orchestrator.post_settlement_loss_cooldown import (
    await_post_loss_cooldown,
)
from src.application.services.orchestrator.post_settlement_resilience import clear_post_settlement_polling_state
from src.application.services.orchestrator.session_target_bootstrap import clear_current_session_redis_keys
from src.application.services.orchestrator.settlement_logic import check_session_limits_before_post_settlement
from src.application.services.orchestrator.settlement_queue_ops import get_redis_client
from src.application.services.orchestrator.settlement_utils import (
    clear_contract_metadata,
    prune_orphan_contract_ids,
)
from src.application.services.settle_log_dedupe import log_settle_info_if_changed
from src.presentation.terminal.settle_log import SETTLE_TOLERANCE


_MAX_POST_SETTLEMENT_CYCLE_ATTEMPTS = 32


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


def _ensure_trading_slot_poll(orch: Any, *, wait_limit: float) -> None:
    """Dispara polling atomico do slot de ciclo sem bloquear a corrotina chamadora."""
    task = getattr(orch, "_trading_slot_poll_task", None)
    if task is not None and not task.done():
        return
    orch._trading_slot_poll_task = asyncio.ensure_future(_release_stuck_trading_slot(orch, wait_limit=wait_limit))


async def _release_stuck_trading_slot(orch: Any, *, wait_limit: float) -> None:
    """Libera is_trading preso apos deadline sem log bloqueante no hot path."""
    poll = 0.05
    deadline = time.monotonic() + float(wait_limit)
    while orch.running and orch.is_trading:
        if time.monotonic() >= deadline:
            orch.is_trading = False
            orch.logger.warning("CICLO: is_trading preso; liberando e retentando")
            return
        await _poll_delay(poll)


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
    """Agenda novo ciclo de decisao logo apos liquidacao do contrato."""
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


async def _try_stop_win_fast_path(orch: Any) -> bool:
    """Curto-circuita pos-liquidacao quando stop-win ja foi atingido."""
    if not check_session_limits_before_post_settlement(orch):
        return False
    await clear_current_session_redis_keys(orch)
    orch.shutdown_reason = "stop_win"
    await orch.state.set_trading(value=False)
    await graceful_shutdown(orch, fast_path=True)
    return True


async def _clean_stale_settlement_and_redis_counters(orch: Any) -> None:
    """Limpa de forma atomica no Redis os contadores e chaves de timeout pendentes."""
    try:
        client = await get_redis_client(orch)
        pipe = client.pipeline()
        pipe.delete("recovery:skip_counter")
        pipe.delete("settlement:queue:priority")
        await pipe.execute()
        log_debug_if_changed(
            orch,
            orch.logger,
            "sre_redis_clean",
            "ok",
            "SRE: Limpeza atomica no Redis concluida para 'recovery:skip_counter' e 'settlement:queue:priority'.",
        )
    except Exception as e:
        orch.logger.error("SRE: Falha ao executar limpeza atomica no Redis: %s", e)


def _record_post_settlement_incomplete(orch: Any) -> None:
    """Registra tentativa incompleta sem forcar deadlock apos 2 ciclos."""
    streak = int(getattr(orch, "_post_settlement_incomplete_streak", 0)) + 1
    orch._post_settlement_incomplete_streak = streak
    log_debug_if_changed(
        orch,
        orch.logger,
        "post_settle_incomplete",
        str(streak),
        "CICLO: pos-liquidacao incompleto | tentativa=%d | janela de tolerancia ativa",
        streak,
    )
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_clean_stale_settlement_and_redis_counters(orch))
    except RuntimeError:
        pass


async def _attempt_post_settlement_trading_cycle(orch: Any, orch_cfg: dict) -> bool:
    """Executa um ciclo pos-liquidacao; True quando o cluster foi executado."""
    orch._dl_fast_cycle = True
    cycle_timeout = float(orch_cfg.get("post_settlement_cycle_timeout_seconds", 90.0))
    try:
        try:
            await asyncio.wait_for(
                orch._run_trading_cycle_if_ready(),
                timeout=cycle_timeout,
            )
            return bool(getattr(orch, "_last_cycle_cluster_executed", False))
        except TimeoutError:
            orch.is_trading = False
            orch.logger.debug("CICLO: timeout pos-liquidacao (%.0fs); backoff na janela de tolerancia", cycle_timeout)
            return False
    finally:
        orch._dl_fast_cycle = False


async def _await_exec_empty_signature_alignment(orch: Any, poll: float) -> None:
    """Apos EXEC_EMPTY em recovery, espera fronteira de 60s; senao polling curto."""
    rm = getattr(orch, "risk_manager", None)
    pending = 0.0
    if rm is not None:
        pending_fn = getattr(rm, "pending_loss_total", None)
        if callable(pending_fn):
            pending = float(pending_fn())
        else:
            pending_map = getattr(rm, "pending_loss", {}) or {}
            if isinstance(pending_map, dict):
                pending = float(sum(float(v) for v in pending_map.values()))
    if pending <= 0.0:
        await _poll_delay(poll)
        return
    cooldown = float(getattr(orch, "_cooldown_until", 0.0))
    now = time.time()
    if cooldown > now:
        await _await_post_settlement_breath(orch, cooldown - now, poll)
        return
    delay = float(seconds_until_next_signature_boundary(orch))
    if delay > 0.0:
        orch_cfg = orch.config.get("orchestrator") if isinstance(getattr(orch, "config", None), dict) else {}
        if not isinstance(orch_cfg, dict):
            orch_cfg = {}
        try:
            empty_cap = float(orch_cfg.get("exec_empty_retry_seconds", 45))
        except (TypeError, ValueError):
            empty_cap = 45.0
        delay = min(delay, max(15.0, empty_cap))
        orch._cooldown_until = now + delay
        await _await_post_settlement_breath(orch, delay, poll)
        return
    await _poll_delay(poll)


async def _apply_tolerance_window_recovery(orch: Any, *, window: float, failed_attempts: int) -> int:
    """Roda orphan cleaner ao esgotar a janela e reinicia contadores sem deadlock."""
    settled = await SettlementOrphanCleaner(orch).reconcile_stale_contracts()
    _record_post_settlement_incomplete(orch)
    orch._post_settlement_deadlock = False
    orch._post_settlement_incomplete_streak = 0
    clear_post_settlement_polling_state(orch)
    settle_content = f"{window:.0f}|{int(settled)}"
    log_settle_info_if_changed(
        orch,
        orch.logger,
        SETTLE_TOLERANCE,
        "settle_tolerance",
        settle_content,
        "janela tolerancia %.0fs | orphans=%d | tentativas=%d | estado reconciliado",
        window,
        settled,
        failed_attempts,
    )
    return settled


async def _run_post_settlement_retry_loop(orch: Any, orch_cfg: dict, poll: float) -> None:
    """Retenta ciclo com backoff exponencial dentro da janela de 120s."""
    trading_wait_limit = float(orch_cfg.get("post_settlement_is_trading_wait_seconds", 90.0))
    window = resolve_settlement_tolerance_window(orch, orch_cfg)
    window_deadline = time.monotonic() + window
    failed_attempts = 0
    while orch.running:
        if await _try_stop_win_fast_path(orch):
            return
        if orch.state.active_contracts:
            if time.monotonic() >= window_deadline:
                await _apply_tolerance_window_recovery(orch, window=window, failed_attempts=failed_attempts)
                window_deadline = time.monotonic() + window
                failed_attempts = 0
            await _poll_delay(poll)
            continue
        if orch.is_trading:
            _ensure_trading_slot_poll(orch, wait_limit=trading_wait_limit)
            slot_task = getattr(orch, "_trading_slot_poll_task", None)
            if slot_task is not None and not slot_task.done():
                await asyncio.wait((slot_task,), timeout=0)
            await _poll_delay(poll)
            continue
        if await _attempt_post_settlement_trading_cycle(orch, orch_cfg):
            orch._post_settlement_incomplete_streak = 0
            orch._post_settlement_deadlock = False
            return
        failed_attempts += 1
        if time.monotonic() >= window_deadline or failed_attempts >= _MAX_POST_SETTLEMENT_CYCLE_ATTEMPTS:
            await _apply_tolerance_window_recovery(orch, window=window, failed_attempts=failed_attempts)
            window_deadline = time.monotonic() + window
            failed_attempts = 0
            if not orch.state.active_contracts and await _attempt_post_settlement_trading_cycle(orch, orch_cfg):
                return
            continue
        backoff = next_settlement_backoff_seconds(failed_attempts - 1)
        await _await_post_settlement_breath(orch, backoff, max(poll, 0.05))
        await _await_exec_empty_signature_alignment(orch, poll)


async def run_post_settlement_breath_and_cycle(orch: Any) -> None:
    """Aplica folego pos-liquidacao antes de um novo ciclo."""
    try:
        if await _try_stop_win_fast_path(orch):
            return
        orch_cfg = orch.config.get("orchestrator") if isinstance(getattr(orch, "config", None), dict) else {}
        poll = 0.25
        breath = float(orch_cfg.get("post_settlement_breath_seconds", 8.0))
        await _await_post_settlement_breath(orch, breath, poll)
        await await_post_loss_cooldown(orch)
        await _run_post_settlement_retry_loop(orch, orch_cfg, poll)
    finally:
        current = asyncio.current_task()
        if current is not None:
            _release_post_settlement_task(orch, current)
