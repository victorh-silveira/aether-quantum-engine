"""Resolver de orchestrator.execution.quality_gate a partir de settings."""

from __future__ import annotations

import json
from typing import Any

from aether_paths import repo_path
from src.application.services.settings_knobs import require_float, require_int, require_keys


_QUALITY_GATE_KEYS = (
    "min_adx_threshold",
    "mandatory_min_trade_score",
    "min_direction_margin",
    "min_payoff_edge",
    "inverted_min_score",
    "min_meta_payoff_zscore",
    "regular",
    "starvation",
    "progressive_conviction",
    "recovery_relax",
    "neutral_meta_payoff",
)
_REGULAR_KEYS = ("min_direction_margin", "min_payoff_edge")
_STARVATION_KEYS = (
    "decay_threshold",
    "decay_step",
    "decay_floor",
    "edge_decay_cycles",
    "edge_decay_multiplier",
    "edge_decay_floor",
    "edge_decay_floor_step",
)
_PROGRESSIVE_KEYS = ("skip_step", "reduction", "margin_floor")
_RECOVERY_RELAX_KEYS = (
    "min_linear",
    "margin_floor",
    "edge_floor",
    "full_pending_units",
    "edge_zscore_waiver",
    "session_stake_unit_bankroll_pct",
)
_NEUTRAL_META_KEYS = ("lo", "hi")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Mescla overrides parciais sobre o bloco SSOT sem perder subchaves."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _settings_quality_gate() -> dict[str, Any]:
    """Carrega quality_gate completo de config/settings.json."""
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    orch = full.get("orchestrator") if isinstance(full, dict) else None
    execution = orch.get("execution") if isinstance(orch, dict) else None
    quality = execution.get("quality_gate") if isinstance(execution, dict) else None
    if not isinstance(quality, dict):
        raise ValueError("orchestrator.execution.quality_gate obrigatorio")
    return quality


def resolve_quality_gate_config(exec_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve quality_gate completo; overrides parciais mesclam no SSOT."""
    base = _settings_quality_gate()
    cfg = exec_cfg if isinstance(exec_cfg, dict) else {}
    override = cfg.get("quality_gate") if isinstance(cfg.get("quality_gate"), dict) else cfg
    if isinstance(override, dict) and override and override is not cfg:
        raw_src = _deep_merge(base, override)
    elif isinstance(override, dict) and override and all(k in override for k in _QUALITY_GATE_KEYS):
        raw_src = override
    elif isinstance(override, dict) and override:
        raw_src = _deep_merge(base, override)
    else:
        raw_src = base
    raw = require_keys(raw_src, _QUALITY_GATE_KEYS, "orchestrator.execution.quality_gate")
    regular = require_keys(raw.get("regular"), _REGULAR_KEYS, "orchestrator.execution.quality_gate.regular")
    starvation = require_keys(raw.get("starvation"), _STARVATION_KEYS, "orchestrator.execution.quality_gate.starvation")
    progressive = require_keys(
        raw.get("progressive_conviction"),
        _PROGRESSIVE_KEYS,
        "orchestrator.execution.quality_gate.progressive_conviction",
    )
    recovery_relax = require_keys(
        raw.get("recovery_relax"),
        _RECOVERY_RELAX_KEYS,
        "orchestrator.execution.quality_gate.recovery_relax",
    )
    neutral = require_keys(
        raw.get("neutral_meta_payoff"),
        _NEUTRAL_META_KEYS,
        "orchestrator.execution.quality_gate.neutral_meta_payoff",
    )
    return {
        "min_adx_threshold": require_float(raw, "min_adx_threshold"),
        "mandatory_min_trade_score": require_float(raw, "mandatory_min_trade_score"),
        "min_direction_margin": require_float(raw, "min_direction_margin"),
        "min_payoff_edge": require_float(raw, "min_payoff_edge"),
        "inverted_min_score": require_float(raw, "inverted_min_score"),
        "min_meta_payoff_zscore": require_float(raw, "min_meta_payoff_zscore"),
        "regular": {
            "min_direction_margin": require_float(regular, "min_direction_margin"),
            "min_payoff_edge": require_float(regular, "min_payoff_edge"),
        },
        "starvation": {
            "decay_threshold": require_int(starvation, "decay_threshold"),
            "decay_step": require_float(starvation, "decay_step"),
            "decay_floor": require_float(starvation, "decay_floor"),
            "edge_decay_cycles": require_int(starvation, "edge_decay_cycles"),
            "edge_decay_multiplier": require_float(starvation, "edge_decay_multiplier"),
            "edge_decay_floor": require_float(starvation, "edge_decay_floor"),
            "edge_decay_floor_step": require_float(starvation, "edge_decay_floor_step"),
        },
        "progressive_conviction": {
            "skip_step": require_int(progressive, "skip_step"),
            "reduction": require_float(progressive, "reduction"),
            "margin_floor": require_float(progressive, "margin_floor"),
        },
        "recovery_relax": {
            "min_linear": require_int(recovery_relax, "min_linear"),
            "margin_floor": require_float(recovery_relax, "margin_floor"),
            "edge_floor": require_float(recovery_relax, "edge_floor"),
            "full_pending_units": require_float(recovery_relax, "full_pending_units"),
            "edge_zscore_waiver": require_float(recovery_relax, "edge_zscore_waiver"),
            "session_stake_unit_bankroll_pct": require_float(recovery_relax, "session_stake_unit_bankroll_pct"),
        },
        "neutral_meta_payoff": {
            "lo": require_float(neutral, "lo"),
            "hi": require_float(neutral, "hi"),
        },
    }


def quality_gate_from_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve quality_gate a partir de config completo ou bloco execution."""
    cfg = config if isinstance(config, dict) else {}
    orch = cfg.get("orchestrator") if isinstance(cfg.get("orchestrator"), dict) else {}
    execution = orch.get("execution") if isinstance(orch.get("execution"), dict) else cfg.get("execution")
    if isinstance(execution, dict):
        return resolve_quality_gate_config(execution)
    return resolve_quality_gate_config(None)
