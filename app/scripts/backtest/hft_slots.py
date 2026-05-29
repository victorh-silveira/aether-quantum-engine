"""Utilitarios de cadencia HFT no backtest M15."""

from __future__ import annotations

from typing import Any

from src.domain.risk.entry_cooldown import resolve_entry_cooldown_ticks


M15_BAR_SECONDS = 900


def hft_slots_per_m15_bar(config: dict[str, Any]) -> int:
    orch = config.get("orchestrator", {}) if isinstance(config.get("orchestrator"), dict) else {}
    cycle_iv = max(1, int(orch.get("cycle_interval_seconds", 15)))
    return max(1, M15_BAR_SECONDS // cycle_iv)


def cooldown_slots(config: dict[str, Any], *, slots_per_bar: int, conviction: float = 0.0) -> int:
    risk = config.get("risk_management", {}) if isinstance(config.get("risk_management"), dict) else {}
    ticks = resolve_entry_cooldown_ticks(risk, conviction)
    if ticks <= 0:
        return 0
    return min(slots_per_bar, ticks)
