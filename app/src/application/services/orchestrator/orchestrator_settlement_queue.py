"""Fila assincrona de liquidacoes com janela de tolerancia e orphan cleaner."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.application.services.infra_timing_config import resolve_orchestrator_timing_config
from src.application.services.orchestrator.settlement_backfill import reconcile_single_contract
from src.application.services.orchestrator.settlement_logic import (
    process_contract_settlement,
)
from src.application.services.orchestrator.settlement_queue_ops import (
    process_redis_settlement_queue,
)
from src.application.services.orchestrator.settlement_utils import (
    clear_contract_metadata,
    prune_orphan_contract_ids,
)
from src.application.services.orchestrator.settlement_ws_queries import fetch_portfolio


DEFAULT_SETTLEMENT_BACKOFF_INITIAL_SECONDS = 1.0
DEFAULT_SETTLEMENT_BACKOFF_MAX_SECONDS = 30.0
DEFAULT_ORPHAN_CLEANER_TIMEOUT_SECONDS = 30.0


def resolve_settlement_tolerance_window(
    orch: Any | None = None,
    orch_cfg: dict[str, Any] | None = None,
) -> float:
    """Janela de tolerancia de settlement lida de settings (SSOT)."""
    cfg = orch_cfg if isinstance(orch_cfg, dict) else {}
    if not cfg and orch is not None:
        raw_cfg = getattr(orch, "config", {})
        chunk = raw_cfg.get("orchestrator") if isinstance(raw_cfg, dict) else {}
        cfg = chunk if isinstance(chunk, dict) else {}
    if "settlement_tolerance_window_seconds" in cfg:
        raw = cfg["settlement_tolerance_window_seconds"]
    else:
        raw = resolve_orchestrator_timing_config(cfg if cfg else None)["settlement_tolerance_window_seconds"]
    return max(1.0, float(raw))


def next_settlement_backoff_seconds(
    attempt: int,
    *,
    initial: float = DEFAULT_SETTLEMENT_BACKOFF_INITIAL_SECONDS,
    maximum: float = DEFAULT_SETTLEMENT_BACKOFF_MAX_SECONDS,
) -> float:
    """Backoff exponencial entre retentativas dentro da janela de tolerancia."""
    step = max(0, int(attempt))
    delay = float(initial) * (2**step)
    return max(float(initial), min(float(maximum), delay))


def _known_contract_ids(orch: Any) -> list[int]:
    """Uniao de IDs rastreados no estado local e no risk manager."""
    ids: set[int] = set()
    for raw in getattr(getattr(orch, "state", None), "active_contracts", {}) or {}:
        try:
            ids.add(int(raw))
        except (TypeError, ValueError):
            continue
    risk = getattr(orch, "risk_manager", None)
    if risk is None:
        return sorted(ids)
    for raw in getattr(risk, "active_contract_ids", []) or []:
        try:
            ids.add(int(raw))
        except (TypeError, ValueError):
            continue
    for raw in getattr(risk, "contract_to_symbol", {}) or {}:
        try:
            ids.add(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(ids)


class SettlementOrphanCleaner:
    """Reconcilia contratos orfaos via portfolio / proposal_open_contract apos a janela."""

    def __init__(self, orch: Any):
        self._orch = orch

    async def reconcile_stale_contracts(self, *, timeout: float | None = None) -> int:
        """Consulta portfolio e liquida IDs locais ausentes do estado aberto da API."""
        orch = self._orch
        ws = getattr(orch, "ws", None)
        if ws is None or not bool(getattr(ws, "is_running", False)):
            return 0
        wait = float(timeout) if timeout is not None else DEFAULT_ORPHAN_CLEANER_TIMEOUT_SECONDS
        known = _known_contract_ids(orch)
        if not known:
            self._prune_local_orphans()
            return 0
        try:
            portfolio = await fetch_portfolio(ws, timeout=wait)
        except Exception as exc:
            orch.logger.info("SETTLE.settle_orphan: orphan cleaner portfolio indisponivel (%s)", type(exc).__name__)
            return 0
        open_ids = {int(row.get("contract_id")) for row in portfolio if row.get("contract_id") is not None}
        settled = 0
        for c_id in known:
            if c_id in open_ids:
                continue
            try:
                if await reconcile_single_contract(orch, c_id):
                    settled += 1
            except Exception as exc:
                orch.logger.info("SETTLE.settle_orphan: orphan cleaner cid=%d (%s)", c_id, type(exc).__name__)
        self._prune_local_orphans()
        if settled > 0:
            orch.logger.info("SETTLE.settle_orphan: orphan cleaner reconciliou %d contrato(s)", settled)
        return settled

    async def passive_reconcile(self, *, timeout: float | None = None) -> bool:
        """Reconciliacao passiva: portfolio vs estado local/Redis sem reiniciar o loop."""
        orch = self._orch
        await process_redis_settlement_queue(orch)
        settled = await self.reconcile_stale_contracts(timeout=timeout)
        remaining = _known_contract_ids(orch)
        clear = len(remaining) == 0 and not bool(getattr(getattr(orch, "state", None), "active_contracts", {}) or {})
        orch.logger.info(
            "SETTLE.settle_reconcile: reconciliacao passiva | orphans=%d | clear=%s | remaining=%d",
            settled,
            clear,
            len(remaining),
        )
        return clear

    def _prune_local_orphans(self) -> None:
        """Remove metadados de risco quando o estado ativo local ja esta vazio."""
        orch = self._orch
        risk = getattr(orch, "risk_manager", None)
        state = getattr(orch, "state", None)
        if risk is None or state is None:
            return
        if getattr(state, "active_contracts", None) and state.active_contracts:
            return
        active_ids = list(getattr(risk, "active_contract_ids", []) or [])
        if not active_ids:
            return
        kept, orphans = prune_orphan_contract_ids(active_ids, getattr(state, "active_contracts", {}) or {})
        risk.active_contract_ids = kept
        if orphans:
            clear_contract_metadata(orphans, risk)


async def start_settlement_worker(orch: Any) -> None:
    """Inicia worker que consome liquidacoes de contrato em background."""
    task = getattr(orch, "_settlement_worker_task", None)
    if task is not None and not task.done():
        return
    orch._settlement_queue = asyncio.Queue()
    orch._settlement_pending_since = float(getattr(orch, "_settlement_pending_since", 0.0) or 0.0)
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
    if float(getattr(orch, "_settlement_pending_since", 0.0) or 0.0) <= 0.0:
        orch._settlement_pending_since = time.monotonic()
    await queue.put(data)


async def _maybe_run_orphan_cleaner(orch: Any) -> None:
    """Dispara orphan cleaner quando a fila/contratos ultrapassam a janela de 120s."""
    known = _known_contract_ids(orch)
    pending_since = float(getattr(orch, "_settlement_pending_since", 0.0) or 0.0)
    if not known:
        orch._settlement_pending_since = 0.0
        return
    if pending_since <= 0.0:
        orch._settlement_pending_since = time.monotonic()
        return
    window = resolve_settlement_tolerance_window(orch)
    if time.monotonic() - pending_since < window:
        return
    settled = await SettlementOrphanCleaner(orch).reconcile_stale_contracts()
    orch._settlement_pending_since = time.monotonic() if _known_contract_ids(orch) else 0.0
    if settled <= 0 and not _known_contract_ids(orch):
        orch._settlement_pending_since = 0.0


async def _settlement_worker_loop(orch: Any) -> None:
    """Consome fila de settlements ate o motor encerrar."""
    queue: asyncio.Queue = orch._settlement_queue
    while orch.running or not queue.empty():
        await process_redis_settlement_queue(orch)
        await _maybe_run_orphan_cleaner(orch)
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=0.25)
        except TimeoutError:
            continue
        try:
            await process_contract_settlement(orch, payload)
        finally:
            queue.task_done()
            if queue.empty() and not _known_contract_ids(orch):
                orch._settlement_pending_since = 0.0
