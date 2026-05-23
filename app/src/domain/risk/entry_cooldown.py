"""Cooldown entre entradas com reducao opcional por conviccao alta."""

from __future__ import annotations

from typing import Any


def resolve_entry_cooldown_ticks(risk_management: dict[str, Any] | None, conviction: float = 0.0) -> int:
    """Retorna ticks de cooldown; menor quando conviccao >= limiar configurado."""
    risk = risk_management if isinstance(risk_management, dict) else {}
    params = risk.get("params", {}) if isinstance(risk.get("params"), dict) else {}
    base = max(0, int(params.get("entry_cooldown_ticks", 0)))
    if base <= 0:
        return 0
    threshold = float(params.get("high_conviction_cooldown_threshold", 0.85))
    reduced = params.get("entry_cooldown_ticks_high_conviction")
    if reduced is None:
        return base
    reduced_n = max(0, int(reduced))
    if float(conviction) >= threshold:
        return min(base, reduced_n)
    return base
