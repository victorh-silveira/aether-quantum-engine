"""Válvula de escape por inanição cronológica no quality gate."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from src.application.services.log_dedupe import LogDeduper, log_info_if_changed


REDIS_SKIPPED_CYCLES_COUNTER_KEY = "state:risk:skipped_cycles_counter"
STARVATION_DECAY_THRESHOLD = 6
STARVATION_DECAY_STEP = 0.10
STARVATION_DECAY_FLOOR = 0.20
PROGRESSIVE_CONVICTION_SKIP_STEP = 5
PROGRESSIVE_CONVICTION_REDUCTION = 0.20
PROGRESSIVE_CONVICTION_MARGIN_FLOOR = 0.0
_STARVATION_ESCAPE_LOG_PREFIX = (
    "[AETHER] EXECUTION_FLOW | Válvula de inanição ativa. "
    "Limite mitigado por decaimento temporal para min {min_direction_margin:.4f} | skipped_cycles={counter}"
)
_PROGRESSIVE_CONVICTION_LOG_PREFIX = (
    "[AETHER] EXECUTION_FLOW | Gatilho de Convicção Progressiva. "
    "min_direction_margin={min_direction_margin:.4f} | factor={factor:.3f} | skipped_cycles={counter}"
)


def starvation_decay_factor(skipped_cycles: int) -> float:
    """Retorna fator multiplicativo de atenuação quando inanição excede o limiar."""
    count = max(0, int(skipped_cycles))
    if count < STARVATION_DECAY_THRESHOLD:
        return 1.0
    return max(
        STARVATION_DECAY_FLOOR,
        1.0 - ((count - (STARVATION_DECAY_THRESHOLD - 1)) * STARVATION_DECAY_STEP),
    )


def progressive_conviction_factor(skipped_cycles: int, *, recovery_active: bool) -> float:
    """Em recovery, reduz 20% o piso a cada 5 ciclos de inanição."""
    if not recovery_active:
        return 1.0
    steps = max(0, int(skipped_cycles)) // PROGRESSIVE_CONVICTION_SKIP_STEP
    if steps <= 0:
        return 1.0
    return (1.0 - PROGRESSIVE_CONVICTION_REDUCTION) ** steps


def apply_progressive_conviction_margin(
    margin: float,
    skipped_cycles: int,
    *,
    recovery_active: bool,
    orch: Any | None = None,
) -> tuple[float, float]:
    """Aplica Gatilho de Convicção Progressiva sobre min_direction_margin em recovery."""
    factor = progressive_conviction_factor(skipped_cycles, recovery_active=recovery_active)
    if factor >= 1.0:
        return float(margin), factor
    mitigated = max(PROGRESSIVE_CONVICTION_MARGIN_FLOOR, float(margin) * factor)
    if orch is not None:
        logger = getattr(orch, "logger", None)
        if logger is not None:
            message = _PROGRESSIVE_CONVICTION_LOG_PREFIX.format(
                min_direction_margin=mitigated,
                factor=factor,
                counter=int(skipped_cycles),
            )
            channel = f"progressive_conviction:{int(skipped_cycles) // PROGRESSIVE_CONVICTION_SKIP_STEP}"
            log_info_if_changed(orch, logger, channel, message, "%s", message)
    return mitigated, factor


def apply_starvation_margin_decay(
    margin: float,
    skipped_cycles: int,
    *,
    orch: Any | None = None,
) -> tuple[float, float]:
    """Atenua piso de margem apos inanicao prolongada ate liberar sinais fracos."""
    decay = starvation_decay_factor(skipped_cycles)
    if decay >= 1.0:
        return float(margin), decay
    mitigated = max(0.0, float(margin) * decay)
    if orch is not None:
        logger = getattr(orch, "logger", None)
        if logger is not None:
            LogDeduper(orch).log_quality_starvation_escape(
                logger,
                skipped_cycles=int(skipped_cycles),
                min_margin=mitigated,
            )
    return mitigated, decay


def apply_starvation_edge_decay(
    edge: float,
    skipped_cycles: int,
) -> float:
    """Atenua piso de payoff previsto após inanição cronológica, aplicando desvio linear a partir de 15 ciclos."""
    decay = starvation_decay_factor(skipped_cycles)
    if decay >= 1.0:
        return float(edge)
    decayed = float(edge) - (1.0 - decay) * 2.0
    floor = 0.01
    if skipped_cycles >= 15:
        floor -= (skipped_cycles - 14) * 0.05
    return max(floor, decayed)


async def load_quality_skipped_cycles_counter(store: Any) -> int:
    """Le contador de ciclos pulados pelo quality guard no StateStore."""
    if store is None or not hasattr(store, "get_string"):
        return 0
    try:
        raw = await store.get_string(REDIS_SKIPPED_CYCLES_COUNTER_KEY)
        if raw is None or not str(raw).strip():
            return 0
        return max(0, int(str(raw).strip()))
    except (TypeError, ValueError, AttributeError):
        return 0


async def reset_quality_skipped_cycles_counter(store: Any) -> None:
    """Zera contador de inanição do quality guard no StateStore."""
    if store is not None and hasattr(store, "set_string"):
        with contextlib.suppress(TypeError, AttributeError):
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
            with contextlib.suppress(TypeError, AttributeError):
                await store.set_string(REDIS_SKIPPED_CYCLES_COUNTER_KEY, str(value))

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_persist())
    except RuntimeError:  # pragma: no cover
        return
