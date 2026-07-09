"""Yield cooperativo apos rejeicao silenciosa do quality gate meta-regressor."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.application.services.orchestrator.orchestrator_data_signature import resolve_signature_boundary_seconds


def _cycle_cadence_seconds(orch: Any) -> int:
    """Intervalo alvo entre ciclos de decisao em segundos."""
    config = getattr(orch, "config", {})
    chunk = config.get("orchestrator") if isinstance(config, dict) else {}
    if not isinstance(chunk, dict):
        return 0
    return int(chunk.get("cycle_interval_seconds") or 0)


def quality_skip_yield_seconds(orch: Any) -> float:
    """Calcula pausa ate a proxima fronteira temporal operacional."""
    boundary = max(60, int(resolve_signature_boundary_seconds(orch)))
    now = time.time()
    next_boundary = (int(now) // boundary + 1) * boundary
    delay = max(0.0, float(next_boundary) - now)
    cadence = _cycle_cadence_seconds(orch)
    if cadence > 0:
        last_end = float(getattr(orch, "_last_cluster_cycle_end", 0.0))
        elapsed = now - last_end if last_end > 0.0 else 0.0
        cadence_remaining = max(0.0, float(cadence) - elapsed)
        delay = max(delay, cadence_remaining)
    return delay


def sanitize_quality_skip_decisions(decisions: dict) -> None:
    """Remove metadados repetitivos do payload apos rejeicao silenciosa."""
    if not isinstance(decisions, dict):
        return
    for entry in decisions.values():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        metrics.pop("quality_guard_reject", None)
        metrics.pop("quality_gate_reason", None)
        metrics.pop("regime_skip_cycle", None)


async def await_quality_skip_yield(orch: Any) -> float:
    """Cede o loop e aguarda a proxima janela temporal limpa."""
    delay = quality_skip_yield_seconds(orch)
    if delay > 0.0:
        await asyncio.sleep(delay)
    return delay
