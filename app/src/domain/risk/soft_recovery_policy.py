"""Politica parametrica do Soft Recovery Adaptativo lida de risk_management."""

from __future__ import annotations

from typing import Any

from src.domain.risk.soft_recovery_config import (
    load_soft_recovery_from_settings,
    pending_waives_scale_explore,
    require_soft_recovery,
    reset_soft_recovery_config_cache,
    resolve_soft_recovery_config,
    soft_cfg,
)


def soft_recovery_enabled(
    engine_config: dict[str, Any] | None = None,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> bool:
    """Indica se soft recovery esta habilitado no engine ou no SSOT."""
    if isinstance(soft_recovery, dict) and ("enabled" in soft_recovery or soft_recovery):
        if "enabled" in soft_recovery:
            return bool(soft_recovery["enabled"])
        return True
    cfg = engine_config if isinstance(engine_config, dict) else {}
    nested = cfg.get("soft_recovery")
    if isinstance(nested, dict) and ("enabled" in nested or nested):
        return bool(nested["enabled"]) if "enabled" in nested else True
    if "dlambert_enabled" in cfg:
        return bool(cfg["dlambert_enabled"])
    return bool(load_soft_recovery_from_settings()["enabled"])


def resolve_amort_cycles(consecutive_losses: int, soft_recovery: dict[str, Any] | None = None) -> int:
    """Calcula ciclos de amortizacao entre amin e amax conforme perdas."""
    cfg = soft_cfg(soft_recovery)
    amin = max(1, int(cfg["amort_cycles_min"]))
    amax = max(amin, int(cfg["amort_cycles_max"]))
    span = amax - amin
    losses = max(0, int(consecutive_losses))
    cycles = amax - min(losses, span)
    res = max(amin, min(amax, cycles))
    return res


def is_recovery_infeasible(
    pending_total: float,
    max_safe_cap: float,
    payout: float,
    soft_recovery: dict[str, Any] | None = None,
) -> bool:
    """True quando a divida pendente nao cabe no horizonte maximo de amortizacao."""
    cfg = soft_cfg(soft_recovery)
    amax = max(1, int(cfg["amort_cycles_max"]))
    cap = float(max_safe_cap)
    pay = float(payout)
    pending = float(pending_total)
    cover_mult = max(1.0, float(cfg.get("cover_multiple", 1.0)))
    if pending <= 0.0:
        return False
    if cap <= 0.0 or pay <= 0.0:
        return True
    return (pending * cover_mult / (cap * pay)) > float(amax)


def configured_max_safe_stake_cap(soft_recovery: dict[str, Any] | None) -> float | None:
    """Retorna teto absoluto de stake seguro quando configurado e positivo."""
    if not isinstance(soft_recovery, dict):
        return None
    if "max_safe_stake_cap" not in soft_recovery:
        return None
    try:
        value = float(soft_recovery["max_safe_stake_cap"])
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def configured_max_safe_stake_pct(soft_recovery: dict[str, Any] | None) -> float:
    """Retorna fracao maxima de bankroll para stake seguro."""
    cfg = soft_cfg(soft_recovery)
    try:
        value = float(cfg["max_safe_stake_pct"])
    except (TypeError, ValueError):
        value = float(load_soft_recovery_from_settings()["max_safe_stake_pct"])
    if value <= 0.0:
        return float(load_soft_recovery_from_settings()["max_safe_stake_pct"])
    return min(value, 1.0)


def fixed_step_progression_multiplier(
    consecutive_losses: int,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> float | None:
    """Multiplicador de progressao em janela linear de perdas consecutivas."""
    cfg = soft_cfg(soft_recovery)
    lo = int(cfg["fixed_step_linear_min"])
    hi = int(cfg["fixed_step_linear_max"])
    premium = float(cfg["fixed_step_unit_premium"])
    losses = max(0, int(consecutive_losses))
    if lo <= losses <= hi:
        return 1.0 + premium
    return None


def apply_small_account_hard_floor(
    cap: float,
    bankroll: float,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> float:
    """Aplica piso duro de stake em contas abaixo do limiar configurado."""
    cfg = soft_cfg(soft_recovery)
    threshold = float(cfg["small_account_hard_floor_threshold"])
    pct = float(cfg["small_account_hard_floor_pct"])
    bal = max(0.0, float(bankroll))
    if bal <= 0.0 or bal >= threshold:
        return float(cap)
    return min(float(cap), bal * pct)


def is_micro_residual_liability(
    bankroll: float,
    pending_total: float,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> bool:
    """True quando a divida pendente e residual relativa ao bankroll micro."""
    bal = float(bankroll)
    pending = float(pending_total)
    if bal <= 0.0 or pending <= 0.0:
        return False
    cfg = soft_cfg(soft_recovery)
    bankroll_max = float(cfg["micro_residual_bankroll_max"])
    pending_max = float(cfg["micro_residual_pending_max"])
    pending_pct = float(cfg["micro_residual_pending_pct"])
    if bal > bankroll_max:
        return False
    if pending > pending_max:
        return False
    return pending < (pending_pct * bal)


def is_low_intensity_recovery(
    bankroll: float,
    pending_total: float,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> bool:
    """Alias semantico de residual micro para intensity baixa de recovery."""
    return is_micro_residual_liability(bankroll, pending_total, soft_recovery=soft_recovery)


def resolve_negative_zscore_veto_floor(
    bankroll: float,
    pending_total: float,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> float:
    """Piso de veto por z-score negativo, relaxado em residual micro."""
    cfg = soft_cfg(soft_recovery)
    if is_micro_residual_liability(bankroll, pending_total, soft_recovery=cfg):
        return float(cfg["micro_residual_zscore_floor"])
    return float(cfg["negative_zscore_veto"])


def resolve_gbdt_waiver_skip_threshold(
    bankroll: float,
    pending_total: float,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> int:
    """Limiar de skips GBDT waiver conforme intensidade da divida."""
    cfg = soft_cfg(soft_recovery)
    if is_micro_residual_liability(bankroll, pending_total, soft_recovery=cfg):
        return max(1, int(cfg["micro_residual_gbdt_waiver_skips"]))
    return max(1, int(cfg["gbdt_waiver_skip_cycles"]))


def cointegration_valve_suppressed(
    bankroll: float,
    pending_total: float,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> bool:
    """Suprime valvula de cointegracao em recovery de baixa intensidade."""
    return is_low_intensity_recovery(bankroll, pending_total, soft_recovery=soft_recovery)


def risk_session_bankroll_pending(risk_manager: Any | None) -> tuple[float, float, dict[str, Any] | None]:
    """Extrai bankroll, pending e soft_recovery do risk manager da sessao."""
    if risk_manager is None:
        return 0.0, 0.0, None
    bankroll = float(getattr(risk_manager, "initial_bankroll", 0.0) or 0.0)
    pending_fn = getattr(risk_manager, "pending_loss_total", None)
    if callable(pending_fn):
        pending = float(pending_fn())
    else:
        pending_map = getattr(risk_manager, "pending_loss", {}) or {}
        pending = sum(float(v) for v in pending_map.values()) if isinstance(pending_map, dict) else 0.0
    soft = getattr(risk_manager, "soft_recovery_config", None)
    return bankroll, pending, soft if isinstance(soft, dict) else None


def negative_zscore_veto_floor_for_risk(risk_manager: Any | None) -> float:
    """Piso de veto z-score a partir do estado do risk manager."""
    bankroll, pending, soft = risk_session_bankroll_pending(risk_manager)
    if bankroll <= 0.0:
        return float(load_soft_recovery_from_settings()["negative_zscore_veto"])
    return resolve_negative_zscore_veto_floor(bankroll, pending, soft_recovery=soft)


def gbdt_waiver_skip_threshold_for_risk(risk_manager: Any | None) -> int:
    """Limiar GBDT waiver a partir do estado do risk manager."""
    bankroll, pending, soft = risk_session_bankroll_pending(risk_manager)
    if bankroll <= 0.0:
        return int(load_soft_recovery_from_settings()["gbdt_waiver_skip_cycles"])
    return resolve_gbdt_waiver_skip_threshold(bankroll, pending, soft_recovery=soft)


__all__ = (
    "apply_small_account_hard_floor",
    "cointegration_valve_suppressed",
    "configured_max_safe_stake_cap",
    "configured_max_safe_stake_pct",
    "fixed_step_progression_multiplier",
    "gbdt_waiver_skip_threshold_for_risk",
    "is_low_intensity_recovery",
    "is_micro_residual_liability",
    "is_recovery_infeasible",
    "load_soft_recovery_from_settings",
    "negative_zscore_veto_floor_for_risk",
    "pending_waives_scale_explore",
    "require_soft_recovery",
    "reset_soft_recovery_config_cache",
    "resolve_amort_cycles",
    "resolve_gbdt_waiver_skip_threshold",
    "resolve_negative_zscore_veto_floor",
    "resolve_soft_recovery_config",
    "risk_session_bankroll_pending",
    "soft_cfg",
    "soft_recovery_enabled",
)
