"""Valvula de escape por inanicao cronologica no quality gate."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from src.application.services.execution_runtime_config import resolve_quality_gate_from_exec
from src.application.services.log_dedupe import LogDeduper, log_info_if_changed


REDIS_SKIPPED_CYCLES_COUNTER_KEY = "state:risk:skipped_cycles_counter"
_STARVATION_ESCAPE_LOG_PREFIX = (
    "[AETHER] EXECUTION_FLOW | Válvula de inanição ativa. "
    "Limite mitigado por decaimento temporal para min {min_direction_margin:.4f} | skipped_cycles={counter}"
)
_PROGRESSIVE_CONVICTION_LOG_PREFIX = (
    "[AETHER] EXECUTION_FLOW | Gatilho de Convicção Progressiva. "
    "min_direction_margin={min_direction_margin:.4f} | factor={factor:.3f} | skipped_cycles={counter}"
)


def _qg(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica  qg."""
    return resolve_quality_gate_from_exec(exec_cfg)


def starvation_decay_factor(skipped_cycles: int, *, exec_cfg: dict[str, Any] | None = None) -> float:
    """Resolve ou aplica starvation decay factor."""
    starvation = _qg(exec_cfg)["starvation"]
    count = max(0, int(skipped_cycles))
    threshold = int(starvation["decay_threshold"])
    if count < threshold:
        return 1.0
    return max(
        float(starvation["decay_floor"]),
        1.0 - ((count - (threshold - 1)) * float(starvation["decay_step"])),
    )


def progressive_conviction_factor(
    skipped_cycles: int,
    *,
    recovery_active: bool,
    exec_cfg: dict[str, Any] | None = None,
) -> float:
    """Resolve ou aplica progressive conviction factor."""
    if not recovery_active:
        return 1.0
    progressive = _qg(exec_cfg)["progressive_conviction"]
    steps = max(0, int(skipped_cycles)) // int(progressive["skip_step"])
    if steps <= 0:
        return 1.0
    return (1.0 - float(progressive["reduction"])) ** steps


def apply_progressive_conviction_margin(
    margin: float,
    skipped_cycles: int,
    *,
    recovery_active: bool,
    orch: Any | None = None,
    exec_cfg: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Resolve ou aplica apply progressive conviction margin."""
    cfg = exec_cfg
    if cfg is None and orch is not None:
        config = getattr(orch, "config", None)
        if isinstance(config, dict):
            orch_cfg = config.get("orchestrator") if isinstance(config.get("orchestrator"), dict) else {}
            cfg = orch_cfg.get("execution") if isinstance(orch_cfg, dict) else None
    progressive = _qg(cfg)["progressive_conviction"]
    factor = progressive_conviction_factor(skipped_cycles, recovery_active=recovery_active, exec_cfg=cfg)
    if factor >= 1.0:
        return float(margin), factor
    mitigated = max(float(progressive["margin_floor"]), float(margin) * factor)
    if orch is not None:
        logger = getattr(orch, "logger", None)
        if logger is not None:
            message = _PROGRESSIVE_CONVICTION_LOG_PREFIX.format(
                min_direction_margin=mitigated,
                factor=factor,
                counter=int(skipped_cycles),
            )
            channel = f"progressive_conviction:{int(skipped_cycles) // int(progressive['skip_step'])}"
            log_info_if_changed(orch, logger, channel, message, "%s", message)
    return mitigated, factor


def apply_starvation_margin_decay(
    margin: float,
    skipped_cycles: int,
    *,
    orch: Any | None = None,
    exec_cfg: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Resolve ou aplica apply starvation margin decay."""
    cfg = exec_cfg
    if cfg is None and orch is not None:
        config = getattr(orch, "config", None)
        if isinstance(config, dict):
            orch_cfg = config.get("orchestrator") if isinstance(config.get("orchestrator"), dict) else {}
            cfg = orch_cfg.get("execution") if isinstance(orch_cfg, dict) else None
    decay = starvation_decay_factor(skipped_cycles, exec_cfg=cfg)
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
    *,
    exec_cfg: dict[str, Any] | None = None,
) -> float:
    """Resolve ou aplica apply starvation edge decay."""
    starvation = _qg(exec_cfg)["starvation"]
    decay = starvation_decay_factor(skipped_cycles, exec_cfg=exec_cfg)
    if decay >= 1.0:
        return float(edge)
    decayed = float(edge) - (1.0 - decay) * float(starvation["edge_decay_multiplier"])
    floor = float(starvation["edge_decay_floor"])
    cycles = int(starvation["edge_decay_cycles"])
    if skipped_cycles >= cycles:
        floor -= (skipped_cycles - (cycles - 1)) * float(starvation["edge_decay_floor_step"])
    return max(floor, decayed)


async def load_quality_skipped_cycles_counter(store: Any) -> int:
    """Resolve ou aplica load quality skipped cycles counter."""
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
    """Resolve ou aplica reset quality skipped cycles counter."""
    if store is not None and hasattr(store, "set_string"):
        with contextlib.suppress(TypeError, AttributeError):
            await store.set_string(REDIS_SKIPPED_CYCLES_COUNTER_KEY, "0")


async def reset_quality_skipped_cycles_counter_for_orch(orch: Any) -> None:
    """Resolve ou aplica reset quality skipped cycles counter for orch."""
    await reset_quality_skipped_cycles_counter(getattr(orch, "state_store", None))
    orch._quality_skipped_cycles_counter = 0


async def prepare_quality_skipped_cycles_counter(orch: Any) -> int:
    """Resolve ou aplica prepare quality skipped cycles counter."""
    count = await load_quality_skipped_cycles_counter(getattr(orch, "state_store", None))
    orch._quality_skipped_cycles_counter = count
    return count


def record_quality_guard_cycle_skip(orch: Any) -> int:
    """Resolve ou aplica record quality guard cycle skip."""
    current = int(getattr(orch, "_quality_skipped_cycles_counter", 0))
    next_val = current + 1
    orch._quality_skipped_cycles_counter = next_val
    _schedule_quality_skipped_cycles_persist(orch)
    return next_val


def _schedule_quality_skipped_cycles_persist(orch: Any) -> None:
    """Resolve ou aplica  schedule quality skipped cycles persist."""
    store = getattr(orch, "state_store", None)
    value = int(getattr(orch, "_quality_skipped_cycles_counter", 0))

    async def _persist() -> None:
        """Resolve ou aplica  persist."""
        if store is not None and hasattr(store, "set_string"):
            with contextlib.suppress(TypeError, AttributeError):
                await store.set_string(REDIS_SKIPPED_CYCLES_COUNTER_KEY, str(value))

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_persist())
    except RuntimeError:
        return
