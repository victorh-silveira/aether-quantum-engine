"""Loader fail-closed de risk_management.soft_recovery a partir de settings.json."""

from __future__ import annotations

from typing import Any

from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_int, require_keys


_SOFT_RECOVERY_KEYS = (
    "enabled",
    "max_safe_stake_cap",
    "max_safe_stake_pct",
    "max_safe_stake_pct_linear2",
    "max_safe_stake_pct_linear3",
    "amort_cycles_min",
    "amort_cycles_max",
    "coing_redirect_drawdown_threshold",
    "micro_residual_bankroll_max",
    "micro_residual_pending_max",
    "micro_residual_pending_pct",
    "micro_residual_zscore_floor",
    "negative_zscore_veto",
    "gbdt_waiver_skip_cycles",
    "micro_residual_gbdt_waiver_skips",
    "fixed_step_linear_min",
    "fixed_step_linear_max",
    "fixed_step_unit_premium",
    "small_account_hard_floor_threshold",
    "small_account_hard_floor_pct",
    "dust_pending_clear_max",
    "near_stop_win_freeze_pct",
    "material_pending_min",
    "linear_bankroll_pct",
    "cover_multiple",
    "infeasible_force_explore",
    "pending_waives_scale_explore",
    "adapted_force_explore",
    "adapted_force_explore_linear_min",
    "live_evidence_force_explore_linear_min",
    "live_evidence_force_explore_n_min",
    "live_evidence_force_explore_wr_max",
)

_CACHE: dict[str, Any] = {"soft_recovery": None}


def require_soft_recovery(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Valida e normaliza o bloco soft_recovery completo."""
    block = require_keys(raw if isinstance(raw, dict) else None, _SOFT_RECOVERY_KEYS, "risk_management.soft_recovery")
    amin = require_int(block, "amort_cycles_min")
    amax = require_int(block, "amort_cycles_max")
    if amin < 1:
        raise ValueError("risk_management.soft_recovery.amort_cycles_min deve ser >= 1")
    if amax < amin:
        raise ValueError("risk_management.soft_recovery.amort_cycles_max deve ser >= amort_cycles_min")
    return {
        "enabled": require_bool(block, "enabled"),
        "max_safe_stake_cap": require_float(block, "max_safe_stake_cap"),
        "max_safe_stake_pct": require_float(block, "max_safe_stake_pct"),
        "max_safe_stake_pct_linear2": require_float(block, "max_safe_stake_pct_linear2"),
        "max_safe_stake_pct_linear3": require_float(block, "max_safe_stake_pct_linear3"),
        "amort_cycles_min": amin,
        "amort_cycles_max": amax,
        "coing_redirect_drawdown_threshold": require_float(block, "coing_redirect_drawdown_threshold"),
        "micro_residual_bankroll_max": require_float(block, "micro_residual_bankroll_max"),
        "micro_residual_pending_max": require_float(block, "micro_residual_pending_max"),
        "micro_residual_pending_pct": require_float(block, "micro_residual_pending_pct"),
        "micro_residual_zscore_floor": require_float(block, "micro_residual_zscore_floor"),
        "negative_zscore_veto": require_float(block, "negative_zscore_veto"),
        "gbdt_waiver_skip_cycles": require_int(block, "gbdt_waiver_skip_cycles"),
        "micro_residual_gbdt_waiver_skips": require_int(block, "micro_residual_gbdt_waiver_skips"),
        "fixed_step_linear_min": require_int(block, "fixed_step_linear_min"),
        "fixed_step_linear_max": require_int(block, "fixed_step_linear_max"),
        "fixed_step_unit_premium": require_float(block, "fixed_step_unit_premium"),
        "small_account_hard_floor_threshold": require_float(block, "small_account_hard_floor_threshold"),
        "small_account_hard_floor_pct": require_float(block, "small_account_hard_floor_pct"),
        "dust_pending_clear_max": require_float(block, "dust_pending_clear_max"),
        "near_stop_win_freeze_pct": require_float(block, "near_stop_win_freeze_pct"),
        "material_pending_min": require_float(block, "material_pending_min"),
        "linear_bankroll_pct": require_float(block, "linear_bankroll_pct"),
        "cover_multiple": max(1.0, min(5.0, require_float(block, "cover_multiple"))),
        "infeasible_force_explore": require_bool(block, "infeasible_force_explore"),
        "pending_waives_scale_explore": require_bool(block, "pending_waives_scale_explore"),
        "adapted_force_explore": require_bool(block, "adapted_force_explore"),
        "adapted_force_explore_linear_min": require_int(block, "adapted_force_explore_linear_min"),
        "live_evidence_force_explore_linear_min": require_int(block, "live_evidence_force_explore_linear_min"),
        "live_evidence_force_explore_n_min": require_int(block, "live_evidence_force_explore_n_min"),
        "live_evidence_force_explore_wr_max": require_float(block, "live_evidence_force_explore_wr_max"),
    }


def reset_soft_recovery_config_cache() -> None:
    """Limpa cache do soft_recovery carregado de settings."""
    _CACHE["soft_recovery"] = None


def load_soft_recovery_from_settings() -> dict[str, Any]:
    """Carrega soft_recovery completo de config/settings.json."""
    cached = _CACHE.get("soft_recovery")
    if cached is not None:
        return cached
    resolved = require_soft_recovery(merge_settings_block(("risk_management", "soft_recovery"), None))
    _CACHE["soft_recovery"] = resolved
    return resolved


def resolve_soft_recovery_config(risk_management: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve soft_recovery com merge de override parcial sobre o SSOT."""
    cfg = risk_management if isinstance(risk_management, dict) else {}
    soft = cfg.get("soft_recovery") if isinstance(cfg.get("soft_recovery"), dict) else None
    if soft is None and cfg and all(k in cfg for k in ("enabled", "max_safe_stake_cap")):
        soft = cfg
    raw = merge_settings_block(("risk_management", "soft_recovery"), soft)
    return require_soft_recovery(raw)


def soft_cfg(soft_recovery: dict[str, Any] | None) -> dict[str, Any]:
    """Normaliza soft_recovery parcial mesclando no SSOT."""
    if isinstance(soft_recovery, dict) and soft_recovery:
        try:
            return require_soft_recovery(merge_settings_block(("risk_management", "soft_recovery"), soft_recovery))
        except (TypeError, ValueError):
            return load_soft_recovery_from_settings()
    return load_soft_recovery_from_settings()


def pending_waives_scale_explore(
    pending_total: float,
    soft_recovery: dict[str, Any] | None = None,
) -> bool:
    """True quando pending material autoriza soft cover apesar de scale discord/adapt."""
    cfg = soft_cfg(soft_recovery)
    if not bool(cfg["pending_waives_scale_explore"]):
        return False
    return float(pending_total) + 1e-12 >= float(cfg["material_pending_min"])
