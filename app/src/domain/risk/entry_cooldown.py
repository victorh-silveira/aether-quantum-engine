"""Cooldown entre entradas com reducao opcional por conviccao alta."""

from __future__ import annotations

from typing import Any


def resolve_entry_cooldown_seconds(risk_management: dict[str, Any] | None, conviction: float = 0.0) -> float | None:
    """Segundos de pausa entre entradas; sempre None (desativado)."""
    _ = risk_management
    _ = conviction
    return None


def resolve_entry_cooldown_ticks(risk_management: dict[str, Any] | None, conviction: float = 0.0) -> int:
    """Retorna ticks de cooldown; sempre 0 (desativado)."""
    _ = risk_management
    _ = conviction
    return 0
