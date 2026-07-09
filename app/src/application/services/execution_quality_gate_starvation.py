"""Válvula de escape por inanição cronológica no quality gate."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.log_dedupe import LogDeduper


REDIS_SKIPPED_CYCLES_COUNTER_KEY = "state:risk:skipped_cycles_counter"
STARVATION_DECAY_THRESHOLD = 15
STARVATION_DECAY_STEP = 0.05
STARVATION_DECAY_FLOOR = 0.50
_STARVATION_ESCAPE_LOG_PREFIX = (
    "[AETHER] EXECUTION_FLOW | Válvula de inanição ativa. "
    "Limite mitigado por decaimento temporal para min {min_direction_margin:.4f} | skipped_cycles={counter}"
)


def starvation_decay_factor(skipped_cycles: int) -> float:
    """Retorna fator multiplicativo de atenuação quando inanição excede o limiar."""
    count = max(0, int(skipped_cycles))
    if count < STARVATION_DECAY_THRESHOLD:
        return 1.0
    return max(STARVATION_DECAY_FLOOR, 1.0 - ((count - 14) * STARVATION_DECAY_STEP))


def apply_starvation_margin_decay(
    margin: float,
    skipped_cycles: int,
    *,
    orch: Any | None = None,
) -> tuple[float, float]:
    """Atenua piso de margem após inanição prolongada e emite log deduplicado."""
    decay = starvation_decay_factor(skipped_cycles)
    if decay >= 1.0:
        return float(margin), decay
    mitigated = float(margin) * decay
    if orch is not None:
        logger = getattr(orch, "logger", None)
        if logger is not None:
            LogDeduper(orch).log_quality_starvation_escape(
                logger,
                skipped_cycles=int(skipped_cycles),
                min_margin=mitigated,
            )
    return mitigated, decay


async def load_quality_skipped_cycles_counter(store: Any) -> int:
    """Le contador de ciclos pulados pelo quality guard no StateStore."""
    if store is None or not hasattr(store, "get_string"):
        return 0
    raw = await store.get_string(REDIS_SKIPPED_CYCLES_COUNTER_KEY)
    if raw is None or not str(raw).strip():
        return 0
    try:
        return max(0, int(str(raw).strip()))
    except (TypeError, ValueError):
        return 0


async def reset_quality_skipped_cycles_counter(store: Any) -> None:
    """Zera contador de inanição do quality guard no StateStore."""
    if store is not None and hasattr(store, "set_string"):
        await store.set_string(REDIS_SKIPPED_CYCLES_COUNTER_KEY, "0")


async def reset_quality_skipped_cycles_counter_for_orch(orch: Any) -> None:
    """Zera contador de inanição no store e no cache do orquestrador."""
    await reset_quality_skipped_cycles_counter(getattr(orch, "state_store", None))
    orch._quality_skipped_cycles_counter = 0


async def prepare_quality_skipped_cycles_counter(orch: Any) -> int:
    """Carrega contador de inanição do Redis para o ciclo corrente."""
    count = await load_quality_skipped_cycles_counter(getattr(orch, "state_store", None))
    orch._quality_skipped_cycles_counter = count
    return count


def record_quality_guard_cycle_skip(orch: Any) -> int:
    """Incrementa contador local e agenda persistencia apos descarte do quality guard."""
    current = int(getattr(orch, "_quality_skipped_cycles_counter", 0))
    next_val = current + 1
    orch._quality_skipped_cycles_counter = next_val
    _schedule_quality_skipped_cycles_persist(orch)
    return next_val


def _schedule_quality_skipped_cycles_persist(orch: Any) -> None:
    """Persiste incremento do contador de inanição de forma assíncrona."""
    store = getattr(orch, "state_store", None)
    value = int(getattr(orch, "_quality_skipped_cycles_counter", 0))

    async def _persist() -> None:
        """Grava contador de inanição no Redis após incremento local."""
        if store is not None and hasattr(store, "set_string"):
            await store.set_string(REDIS_SKIPPED_CYCLES_COUNTER_KEY, str(value))

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_persist())
    except RuntimeError:
        return
