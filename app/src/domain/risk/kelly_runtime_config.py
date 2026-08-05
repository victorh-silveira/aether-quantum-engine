"""Resolver de knobs runtime de Kelly a partir de settings."""

from __future__ import annotations

import json
from typing import Any

from aether_paths import repo_path
from src.domain.config_knobs import require_float, require_int, require_keys


_KELLY_RUNTIME_KEYS = (
    "neutral_bankroll_pct",
    "turbo_edge_zscore_threshold",
    "turbo_edge_stake_multiplier",
    "payout_fallback",
    "adaptive_recovery_factor_cap",
    "d_squeeze_sovereign_trade_score",
    "micro_bankroll_threshold",
    "micro_bankroll_pct",
    "turbo_live_n_min",
    "turbo_live_brier_max",
    "fraction",
    "fraction_base_retention",
    "fraction_reference",
    "fraction_compressed",
    "kelly_p_floor",
    "target_damping_floor",
    "target_damping_span",
    "penalty_smoothing_trade_score_min",
    "recovery_sizing_conviction",
    "recovery_min_conviction",
    "recovery_force_pending_min",
    "recovery_min_val_accuracy",
    "recovery_conviction_ladder",
)

_LADDER_KEYS = ("losses_1", "losses_2", "losses_3", "losses_4")

_CACHE: dict[str, Any] = {"kelly_runtime": None}


def resolve_kelly_runtime_config(kelly: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve ou aplica resolve kelly runtime config."""
    raw = require_keys(kelly if isinstance(kelly, dict) else None, _KELLY_RUNTIME_KEYS, "risk_management.kelly")
    ladder = require_keys(
        raw.get("recovery_conviction_ladder"), _LADDER_KEYS, "risk_management.kelly.recovery_conviction_ladder"
    )
    return {
        "neutral_bankroll_pct": require_float(raw, "neutral_bankroll_pct"),
        "turbo_edge_zscore_threshold": require_float(raw, "turbo_edge_zscore_threshold"),
        "turbo_edge_stake_multiplier": require_float(raw, "turbo_edge_stake_multiplier"),
        "payout_fallback": require_float(raw, "payout_fallback"),
        "adaptive_recovery_factor_cap": require_float(raw, "adaptive_recovery_factor_cap"),
        "d_squeeze_sovereign_trade_score": require_float(raw, "d_squeeze_sovereign_trade_score"),
        "micro_bankroll_threshold": require_float(raw, "micro_bankroll_threshold"),
        "micro_bankroll_pct": require_float(raw, "micro_bankroll_pct"),
        "turbo_live_n_min": require_int(raw, "turbo_live_n_min"),
        "turbo_live_brier_max": require_float(raw, "turbo_live_brier_max"),
        "fraction": require_float(raw, "fraction"),
        "fraction_base_retention": require_float(raw, "fraction_base_retention"),
        "fraction_reference": require_float(raw, "fraction_reference"),
        "fraction_compressed": require_float(raw, "fraction_compressed"),
        "kelly_p_floor": max(0.51, min(0.65, require_float(raw, "kelly_p_floor"))),
        "target_damping_floor": require_float(raw, "target_damping_floor"),
        "target_damping_span": require_float(raw, "target_damping_span"),
        "penalty_smoothing_trade_score_min": require_float(raw, "penalty_smoothing_trade_score_min"),
        "recovery_sizing_conviction": require_float(raw, "recovery_sizing_conviction"),
        "recovery_min_conviction": require_float(raw, "recovery_min_conviction"),
        "recovery_force_pending_min": require_float(raw, "recovery_force_pending_min"),
        "recovery_min_val_accuracy": require_float(raw, "recovery_min_val_accuracy"),
        "recovery_conviction_ladder": {k: require_float(ladder, k) for k in _LADDER_KEYS},
    }


def reset_kelly_runtime_config_cache() -> None:
    """Resolve ou aplica reset kelly runtime config cache."""
    _CACHE["kelly_runtime"] = None


def load_kelly_runtime_from_settings() -> dict[str, Any]:
    """Resolve ou aplica load kelly runtime from settings."""
    cached = _CACHE.get("kelly_runtime")
    if cached is not None:
        return cached
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    rm = full.get("risk_management") if isinstance(full, dict) else None
    kelly = rm.get("kelly") if isinstance(rm, dict) else None
    resolved = resolve_kelly_runtime_config(kelly if isinstance(kelly, dict) else None)
    _CACHE["kelly_runtime"] = resolved
    return resolved


def kelly_runtime_from_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve ou aplica kelly runtime from config."""
    cfg = config if isinstance(config, dict) else {}
    kelly = cfg.get("kelly")
    if isinstance(kelly, dict) and all(k in kelly for k in ("neutral_bankroll_pct", "recovery_conviction_ladder")):
        return resolve_kelly_runtime_config(kelly)
    rm = cfg.get("risk_management")
    if isinstance(rm, dict) and isinstance(rm.get("kelly"), dict):
        return resolve_kelly_runtime_config(rm["kelly"])
    return load_kelly_runtime_from_settings()
