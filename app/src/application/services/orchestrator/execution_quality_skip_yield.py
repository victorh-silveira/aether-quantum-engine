"""Compatibilidade de yield pos quality-skip sem reter o loop."""

from __future__ import annotations

from typing import Any


def quality_skip_yield_seconds(orch: Any) -> float:
    """Retorna zero: sem pausa apos rejeicao silenciosa do quality gate."""
    _ = orch
    return 0.0


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
    """Nao retém o loop apos quality skip."""
    _ = orch
    return 0.0
