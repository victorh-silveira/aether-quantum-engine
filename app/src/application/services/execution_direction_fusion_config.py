"""Parse SSOT de knobs fusion_* em orchestrator.execution.scale_vision."""

from __future__ import annotations

from typing import Any

from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_keys


_FUSION_KEYS = (
    "fusion_enabled",
    "fusion_replace_adapt_flip",
    "fusion_w_macro",
    "fusion_w_micro_bar",
    "fusion_w_mini",
    "fusion_w_mili",
    "fusion_w_tape",
    "fusion_meta_ev_weight",
    "fusion_loss_weight",
    "fusion_tcn_shrink_near_half",
    "fusion_block_when_tcn_pos_edge",
    "fusion_min_edge_execute",
    "fusion_weak_ev_soft_kelly_mult",
    "fusion_weak_ev_seed_soft_kelly_mult",
)

__all__ = ("parse_direction_fusion_config",)


def parse_direction_fusion_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve knobs fusion_* em orchestrator.execution.scale_vision."""
    block = require_keys(
        merge_settings_block(
            ("orchestrator", "execution", "scale_vision"),
            raw if isinstance(raw, dict) else None,
        ),
        _FUSION_KEYS,
        "orchestrator.execution.scale_vision",
    )
    shrink = require_float(block, "fusion_tcn_shrink_near_half")
    if shrink < 0.0 or shrink > 1.0:
        raise ValueError("orchestrator.execution.scale_vision.fusion_tcn_shrink_near_half deve estar em [0, 1]")
    weights = {}
    for key in (
        "fusion_w_macro",
        "fusion_w_micro_bar",
        "fusion_w_mini",
        "fusion_w_mili",
        "fusion_w_tape",
        "fusion_meta_ev_weight",
        "fusion_loss_weight",
    ):
        val = require_float(block, key)
        if val < 0.0 or val > 2.0:
            raise ValueError(f"orchestrator.execution.scale_vision.{key} deve estar em [0, 2]")
        weights[key] = val
    min_edge = require_float(block, "fusion_min_edge_execute")
    if min_edge < 0.0 or min_edge > 0.5:
        raise ValueError("orchestrator.execution.scale_vision.fusion_min_edge_execute deve estar em [0, 0.5]")
    weak_soft = require_float(block, "fusion_weak_ev_soft_kelly_mult")
    seed_soft = require_float(block, "fusion_weak_ev_seed_soft_kelly_mult")
    for name, val in (
        ("fusion_weak_ev_soft_kelly_mult", weak_soft),
        ("fusion_weak_ev_seed_soft_kelly_mult", seed_soft),
    ):
        if val <= 0.0 or val > 1.0:
            raise ValueError(f"orchestrator.execution.scale_vision.{name} deve estar em (0, 1]")
    return {
        "fusion_enabled": require_bool(block, "fusion_enabled"),
        "fusion_replace_adapt_flip": require_bool(block, "fusion_replace_adapt_flip"),
        "fusion_tcn_shrink_near_half": shrink,
        "fusion_block_when_tcn_pos_edge": require_bool(block, "fusion_block_when_tcn_pos_edge"),
        "fusion_min_edge_execute": min_edge,
        "fusion_weak_ev_soft_kelly_mult": weak_soft,
        "fusion_weak_ev_seed_soft_kelly_mult": seed_soft,
        **weights,
    }
