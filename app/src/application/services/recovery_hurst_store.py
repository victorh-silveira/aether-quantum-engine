"""Persistencia do contador de SKIPs Hurst em recovery via StateStore."""

from __future__ import annotations

from typing import Any

from src.domain.risk.recovery_hurst_decay import REDIS_SKIP_COUNTER_KEY


async def load_recovery_skip_counter(store: Any) -> int:
    """Le contador de SKIPs Hurst do StateStore."""
    if store is None or not hasattr(store, "get_string"):
        return 0
    raw = await store.get_string(REDIS_SKIP_COUNTER_KEY)
    if raw is None or not str(raw).strip():
        return 0
    try:
        return max(0, int(str(raw).strip()))
    except (TypeError, ValueError):
        return 0


async def increment_recovery_skip_counter(store: Any) -> int:
    """Incrementa contador de SKIPs Hurst e persiste no StateStore."""
    current = await load_recovery_skip_counter(store)
    next_val = current + 1
    if store is not None and hasattr(store, "set_string"):
        await store.set_string(REDIS_SKIP_COUNTER_KEY, str(next_val))
    return next_val


async def reset_recovery_skip_counter(store: Any) -> None:
    """Zera contador de SKIPs Hurst no StateStore."""
    if store is not None and hasattr(store, "set_string"):
        await store.set_string(REDIS_SKIP_COUNTER_KEY, "0")


async def reset_recovery_skip_counter_for_orch(orch) -> None:
    """Zera contador Hurst no store e no cache do orquestrador."""
    await reset_recovery_skip_counter(getattr(orch, "state_store", None))
    orch._recovery_skip_counter = 0


async def prepare_recovery_skip_counter(orch, *, recovery_active: bool) -> int:
    """Carrega ou zera contador Hurst conforme modo recovery."""
    if not recovery_active:
        await reset_recovery_skip_counter_for_orch(orch)
        return 0
    count = await load_recovery_skip_counter(getattr(orch, "state_store", None))
    orch._recovery_skip_counter = count
    return count
