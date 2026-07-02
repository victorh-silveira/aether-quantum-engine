"""Auditoria profit_table com backoff apos instabilidade de rede."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.application.services.orchestrator.settlement_reconciliation import reconcile_after_ws_recovery


logger = logging.getLogger("AETH")
_MAX_ATTEMPTS = 8
_INITIAL_BACKOFF_SEC = 0.5
_MAX_BACKOFF_SEC = 60.0


def _orch_running(orch: Any) -> bool:
    """Indica se o motor continua em execucao."""
    return bool(getattr(orch, "running", False))


async def _profit_table_audit_loop(orch: Any, *, reason: str) -> None:
    """Tenta reconciliar contratos via profit_table ate sucesso ou esgotar tentativas."""
    backoff = _INITIAL_BACKOFF_SEC
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if not _orch_running(orch):
            return
        ws = getattr(orch, "ws", None)
        if ws is None or not getattr(ws, "is_running", False):
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, _MAX_BACKOFF_SEC)
            continue
        try:
            result = await reconcile_after_ws_recovery(orch)
            if not result.errors:
                logger.info(
                    "RECONCILE: profit_table audit ok | reason=%s | settled=%d",
                    reason,
                    result.settled_count,
                )
                return
        except Exception as exc:
            logger.warning(
                "RECONCILE: profit_table audit falhou attempt=%d reason=%s: %s",
                attempt,
                reason,
                exc,
            )
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2.0, _MAX_BACKOFF_SEC)
    logger.warning("RECONCILE: profit_table audit esgotou tentativas | reason=%s", reason)


def schedule_profit_table_audit(orch: Any, *, reason: str = "broker_unavailable") -> None:
    """Agenda auditoria profit_table em background com backoff exponencial."""
    task = getattr(orch, "_profit_table_audit_task", None)
    if isinstance(task, asyncio.Task) and not task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    orch._profit_table_audit_task = loop.create_task(
        _profit_table_audit_loop(orch, reason=str(reason)),
        name="profit-table-audit",
    )
