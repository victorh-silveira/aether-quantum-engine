"""Resolvers de knobs de execucao a partir de settings.json."""

from __future__ import annotations

import json
from typing import Any

from aether_paths import repo_path
from src.domain.config_knobs import require_bool, require_float, require_int, require_keys, require_mapping


_LOSS_DISCONNECT_RULE_KEYS = ("edge_min", "margin_max", "score")
_LOSS_DISCONNECT_SIDE_KEYS = ("edge_min", "calibrated_side_max", "score")
_LOSS_DISCONNECT_RAW_KEYS = ("edge_min", "raw_side_max", "score")
_LOSS_DISCONNECT_Z_KEYS = ("z_min", "margin_max", "score")
_MARKET_RANK_KEYS = (
    "weight_trade_score",
    "weight_edge",
    "weight_margin",
    "weight_meta_z",
    "blend_primary",
    "blend_secondary",
    "execute_bonus",
    "deploy_ok_bonus",
    "last_loss_penalty",
    "recovery_last_loss_penalty",
    "rotate_bonus",
    "adx_weak_penalty",
    "adx_strong_bonus",
    "hurst_trend_bonus",
    "hurst_mean_revert_penalty",
    "live_brier_hard_penalty",
    "live_brier_soft_penalty",
    "live_brier_hard_above",
    "live_brier_soft_above",
    "live_n_min",
    "ece_hard_penalty",
    "ece_soft_penalty",
    "ece_hard_above",
    "ece_soft_above",
    "thin_margin_penalty",
    "thin_margin_below",
    "squeeze_recovery_penalty",
    "indicator_defaults",
)
_EDGE_ZSCORE_KEYS = (
    "window_min",
    "window_default",
    "window_max",
    "win_threshold",
    "std_eps",
    "turbo_threshold",
)
_META_VETO_KEYS = (
    "negative_zscore_threshold",
    "neutral_edge_floor",
    "soft_veto_score_factor",
    "soft_veto_min_score",
    "squeeze_trade_score",
)
_REGIME_KEYS = ("chop_congestion_z_edge", "tick_accel_neutral_eps")
_VOL_BOOST_KEYS = ("mandatory_score", "min_edge")
_FORCE_KEYS = ("min_trade_score", "stake_min_floor", "stake_min_default", "direction_split")
_CROSS_CORR_KEYS = (
    "squeeze_vol_ratio_max",
    "min_margin",
    "dl_raw_weight",
    "high_corr_abs",
    "high_corr_retention_floor",
    "high_corr_retention_coef",
    "low_corr_abs",
    "low_corr_weight_cap",
    "low_corr_weight_boost",
)
_LOSS_PROTECTION_BASE = (
    "min_direction_margin",
    "recovery_min_direction_margin",
    "recovery_min_hurst",
    "max_edge_without_margin",
    "max_zscore_without_margin",
    "disconnect",
)

_CACHE: dict[str, Any] = {"execution": None}


def _load_execution_from_settings() -> dict[str, Any]:
    """Resolve ou aplica  load execution from settings."""
    cached = _CACHE.get("execution")
    if cached is not None:
        return cached
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    orch = full.get("orchestrator") if isinstance(full, dict) else None
    execution = orch.get("execution") if isinstance(orch, dict) else None
    if not isinstance(execution, dict):
        raise ValueError("orchestrator.execution obrigatorio")
    _CACHE["execution"] = execution
    return execution


def reset_execution_runtime_cache() -> None:
    """Resolve ou aplica reset execution runtime cache."""
    _CACHE["execution"] = None


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


def _execution_block(exec_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Carrega orchestrator.execution do SSOT e aplica override parcial."""
    base = _load_execution_from_settings()
    if isinstance(exec_cfg, dict) and exec_cfg:
        return _deep_merge(base, exec_cfg)
    return base


def resolve_loss_protection_config(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica resolve loss protection config."""
    raw = require_mapping(
        _execution_block(exec_cfg), "loss_protection", _LOSS_PROTECTION_BASE, "orchestrator.execution"
    )
    disconnect = raw["disconnect"]
    if not isinstance(disconnect, dict):
        raise ValueError("orchestrator.execution.loss_protection.disconnect obrigatorio")
    for key, keys in (
        ("edge_margin_soft", _LOSS_DISCONNECT_RULE_KEYS),
        ("edge_margin_hard", _LOSS_DISCONNECT_RULE_KEYS),
        ("zscore_margin", _LOSS_DISCONNECT_Z_KEYS),
        ("edge_calibrated_side", _LOSS_DISCONNECT_SIDE_KEYS),
        ("edge_raw_side", _LOSS_DISCONNECT_RAW_KEYS),
    ):
        require_keys(disconnect.get(key), keys, f"orchestrator.execution.loss_protection.disconnect.{key}")
    if "block_threshold" not in disconnect:
        raise ValueError("orchestrator.execution.loss_protection.disconnect.block_threshold obrigatorio")
    return {
        "min_direction_margin": require_float(raw, "min_direction_margin"),
        "recovery_min_direction_margin": require_float(raw, "recovery_min_direction_margin"),
        "recovery_min_hurst": require_float(raw, "recovery_min_hurst"),
        "max_edge_without_margin": require_float(raw, "max_edge_without_margin"),
        "max_zscore_without_margin": require_float(raw, "max_zscore_without_margin"),
        "disconnect": {
            "edge_margin_soft": {
                k: require_float(disconnect["edge_margin_soft"], k) for k in _LOSS_DISCONNECT_RULE_KEYS
            },
            "edge_margin_hard": {
                k: require_float(disconnect["edge_margin_hard"], k) for k in _LOSS_DISCONNECT_RULE_KEYS
            },
            "zscore_margin": {k: require_float(disconnect["zscore_margin"], k) for k in _LOSS_DISCONNECT_Z_KEYS},
            "edge_calibrated_side": {
                k: require_float(disconnect["edge_calibrated_side"], k) for k in _LOSS_DISCONNECT_SIDE_KEYS
            },
            "edge_raw_side": {k: require_float(disconnect["edge_raw_side"], k) for k in _LOSS_DISCONNECT_RAW_KEYS},
            "block_threshold": require_float(disconnect, "block_threshold"),
        },
    }


def resolve_market_rank_composite(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica resolve market rank composite."""
    market_rank = require_mapping(_execution_block(exec_cfg), "market_rank", ("composite",), "orchestrator.execution")
    composite = require_keys(
        market_rank.get("composite"), _MARKET_RANK_KEYS, "orchestrator.execution.market_rank.composite"
    )
    defaults = require_keys(
        composite.get("indicator_defaults"),
        ("adx", "vol_ratio", "hurst"),
        "orchestrator.execution.market_rank.composite.indicator_defaults",
    )
    out = {k: require_float(composite, k) for k in _MARKET_RANK_KEYS if k != "indicator_defaults"}
    out["live_n_min"] = require_int(composite, "live_n_min")
    out["indicator_defaults"] = {k: require_float(defaults, k) for k in ("adx", "vol_ratio", "hurst")}
    return out


def resolve_edge_zscore_runtime(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica resolve edge zscore runtime."""
    raw = require_mapping(_execution_block(exec_cfg), "edge_zscore", _EDGE_ZSCORE_KEYS, "orchestrator.execution")
    return {
        "window_min": require_int(raw, "window_min"),
        "window_default": require_int(raw, "window_default"),
        "window_max": require_int(raw, "window_max"),
        "win_threshold": require_float(raw, "win_threshold"),
        "std_eps": require_float(raw, "std_eps"),
        "turbo_threshold": require_float(raw, "turbo_threshold"),
    }


def resolve_meta_payoff_veto_config(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica resolve meta payoff veto config."""
    raw = require_mapping(_execution_block(exec_cfg), "meta_payoff_veto", _META_VETO_KEYS, "orchestrator.execution")
    return {k: require_float(raw, k) for k in _META_VETO_KEYS}


def resolve_regime_micro_freeze_config(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica resolve regime micro freeze config."""
    raw = require_mapping(_execution_block(exec_cfg), "regime_micro_freeze", _REGIME_KEYS, "orchestrator.execution")
    return {k: require_float(raw, k) for k in _REGIME_KEYS}


def resolve_volatility_booster_config(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica resolve volatility booster config."""
    raw = require_mapping(_execution_block(exec_cfg), "volatility_booster", _VOL_BOOST_KEYS, "orchestrator.execution")
    return {k: require_float(raw, k) for k in _VOL_BOOST_KEYS}


def resolve_force_trade_config(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica resolve force trade config."""
    raw = require_mapping(_execution_block(exec_cfg), "force_trade", _FORCE_KEYS, "orchestrator.execution")
    return {k: require_float(raw, k) for k in _FORCE_KEYS}


def resolve_cross_corr_config(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica resolve cross corr config."""
    raw = require_mapping(_execution_block(exec_cfg), "cross_corr", _CROSS_CORR_KEYS, "orchestrator.execution")
    return {k: require_float(raw, k) for k in _CROSS_CORR_KEYS}


def resolve_side_equilibrium_config(exec_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve config de telemetria side_equilibrium (sem bloqueio de execucao)."""
    keys = (
        "enabled",
        "small_window",
        "large_window",
        "n_min_small",
        "n_min_large",
        "wr_floor_small",
        "wr_floor_large",
        "freq_bias_max_small",
        "freq_bias_max_large",
        "kelly_mult_soft",
        "margin_boost_soft",
        "break_even_wr",
    )
    raw = require_mapping(_execution_block(exec_cfg), "side_equilibrium", keys, "orchestrator.execution")
    return {
        "enabled": require_bool(raw, "enabled"),
        "small_window": require_int(raw, "small_window"),
        "large_window": require_int(raw, "large_window"),
        "n_min_small": require_int(raw, "n_min_small"),
        "n_min_large": require_int(raw, "n_min_large"),
        "wr_floor_small": require_float(raw, "wr_floor_small"),
        "wr_floor_large": require_float(raw, "wr_floor_large"),
        "freq_bias_max_small": require_float(raw, "freq_bias_max_small"),
        "freq_bias_max_large": require_float(raw, "freq_bias_max_large"),
        "kelly_mult_soft": require_float(raw, "kelly_mult_soft"),
        "margin_boost_soft": require_float(raw, "margin_boost_soft"),
        "break_even_wr": require_float(raw, "break_even_wr"),
    }
