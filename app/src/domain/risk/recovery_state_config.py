"""Resolver de risk_management.recovery_state."""

from __future__ import annotations

import json
from typing import Any

from aether_paths import repo_path
from src.domain.config_knobs import require_float, require_int, require_keys


_RECOVERY_STATE_KEYS = (
    "critical_linear_losses",
    "critical_pending_total",
    "put_extreme_raw_prob",
    "call_extreme_raw_prob",
    "cointegration_drawdown_fraction",
    "micro_bankroll_threshold",
    "micro_tail_linear_level",
    "micro_tail_unit_multiplier",
    "micro_unit_floor",
    "micro_unit_bankroll_pct",
    "raw_prob_default",
)

_CACHE: dict[str, Any] = {"recovery_state": None}


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Resolve ou aplica  normalize."""
    block = require_keys(raw, _RECOVERY_STATE_KEYS, "risk_management.recovery_state")
    return {
        "critical_linear_losses": require_int(block, "critical_linear_losses"),
        "critical_pending_total": require_float(block, "critical_pending_total"),
        "put_extreme_raw_prob": require_float(block, "put_extreme_raw_prob"),
        "call_extreme_raw_prob": require_float(block, "call_extreme_raw_prob"),
        "cointegration_drawdown_fraction": require_float(block, "cointegration_drawdown_fraction"),
        "micro_bankroll_threshold": require_float(block, "micro_bankroll_threshold"),
        "micro_tail_linear_level": require_int(block, "micro_tail_linear_level"),
        "micro_tail_unit_multiplier": require_float(block, "micro_tail_unit_multiplier"),
        "micro_unit_floor": require_float(block, "micro_unit_floor"),
        "micro_unit_bankroll_pct": require_float(block, "micro_unit_bankroll_pct"),
        "raw_prob_default": require_float(block, "raw_prob_default"),
    }


def resolve_recovery_state_config(risk_management: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve ou aplica resolve recovery state config."""
    cfg = risk_management if isinstance(risk_management, dict) else {}
    raw = cfg.get("recovery_state")
    if isinstance(raw, dict) and raw:
        return _normalize(raw)
    return load_recovery_state_from_settings()


def reset_recovery_state_config_cache() -> None:
    """Resolve ou aplica reset recovery state config cache."""
    _CACHE["recovery_state"] = None


def load_recovery_state_from_settings() -> dict[str, Any]:
    """Resolve ou aplica load recovery state from settings."""
    cached = _CACHE.get("recovery_state")
    if cached is not None:
        return cached
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    rm = full.get("risk_management") if isinstance(full, dict) else None
    raw = rm.get("recovery_state") if isinstance(rm, dict) else None
    if not isinstance(raw, dict):
        raise ValueError("risk_management.recovery_state obrigatorio")
    resolved = _normalize(raw)
    _CACHE["recovery_state"] = resolved
    return resolved
