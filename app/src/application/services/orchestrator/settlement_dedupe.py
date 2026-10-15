"""Estado de deduplicacao das fontes de liquidacao concorrentes."""

from __future__ import annotations

import asyncio
from typing import Any


def settlement_lock(orch: Any) -> asyncio.Lock:
    """Serializa fontes concorrentes de liquidacao no mesmo orquestrador."""
    lock = getattr(orch, "_contract_settlement_lock", None)
    if not isinstance(lock, asyncio.Lock):
        lock = asyncio.Lock()
        orch._contract_settlement_lock = lock
    return lock


def settlement_seen(orch: Any) -> set[int]:
    """Retem IDs liquidados durante a sessao, mesmo apos limpeza do cluster."""
    seen = getattr(orch, "_settled_contract_ids", None)
    if not isinstance(seen, set):
        seen = set()
        orch._settled_contract_ids = seen
    return seen
