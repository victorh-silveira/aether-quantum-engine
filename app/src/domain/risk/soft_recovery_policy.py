"""Politica parametrica do Soft Recovery Adaptativo lida de risk_management."""

from __future__ import annotations

from typing import Any


DEFAULT_AMORT_CYCLES_MIN = 2
DEFAULT_AMORT_CYCLES_MAX = 5
DEFAULT_MAX_SAFE_STAKE_CAP = 4.20
DEFAULT_COING_REDIRECT_DRAWDOWN = 15.0
DEFAULT_MICRO_RESIDUAL_BANKROLL_MAX = 250.0
DEFAULT_MICRO_RESIDUAL_PENDING_MAX = 5.0
DEFAULT_MICRO_RESIDUAL_PENDING_PCT = 0.05
DEFAULT_NEGATIVE_ZSCORE_VETO = -0.20
DEFAULT_MICRO_RESIDUAL_ZSCORE_FLOOR = -0.60
DEFAULT_GBDT_WAIVER_SKIP_CYCLES = 30
DEFAULT_MICRO_RESIDUAL_GBDT_WAIVER_SKIPS = 6
FIXED_STEP_LINEAR_MIN = 3
FIXED_STEP_LINEAR_MAX = 4
FIXED_STEP_UNIT_PREMIUM = 0.15
SMALL_ACCOUNT_HARD_FLOOR_THRESHOLD = 100.0
SMALL_ACCOUNT_HARD_FLOOR_PCT = 0.05


def resolve_soft_recovery_config(risk_management: dict[str, Any] | None) -> dict[str, Any]:
    """Extrai bloco soft_recovery com defaults seguros para micro-banca."""
    cfg = risk_management if isinstance(risk_management, dict) else {}
    soft = cfg.get("soft_recovery")
    if not isinstance(soft, dict):
        soft = {}
    return {
        "enabled": bool(soft.get("enabled", True)),
        "max_safe_stake_cap": float(soft.get("max_safe_stake_cap", DEFAULT_MAX_SAFE_STAKE_CAP)),
        "amort_cycles_min": int(soft.get("amort_cycles_min", DEFAULT_AMORT_CYCLES_MIN)),
        "amort_cycles_max": int(soft.get("amort_cycles_max", DEFAULT_AMORT_CYCLES_MAX)),
        "coing_redirect_drawdown_threshold": float(
            soft.get("coing_redirect_drawdown_threshold", DEFAULT_COING_REDIRECT_DRAWDOWN)
        ),
        "micro_residual_bankroll_max": float(
            soft.get("micro_residual_bankroll_max", DEFAULT_MICRO_RESIDUAL_BANKROLL_MAX)
        ),
        "micro_residual_pending_max": float(soft.get("micro_residual_pending_max", DEFAULT_MICRO_RESIDUAL_PENDING_MAX)),
        "micro_residual_pending_pct": float(soft.get("micro_residual_pending_pct", DEFAULT_MICRO_RESIDUAL_PENDING_PCT)),
        "micro_residual_zscore_floor": float(
            soft.get("micro_residual_zscore_floor", DEFAULT_MICRO_RESIDUAL_ZSCORE_FLOOR)
        ),
        "gbdt_waiver_skip_cycles": int(soft.get("gbdt_waiver_skip_cycles", DEFAULT_GBDT_WAIVER_SKIP_CYCLES)),
        "micro_residual_gbdt_waiver_skips": int(
            soft.get("micro_residual_gbdt_waiver_skips", DEFAULT_MICRO_RESIDUAL_GBDT_WAIVER_SKIPS)
        ),
        "fixed_step_linear_min": int(soft.get("fixed_step_linear_min", FIXED_STEP_LINEAR_MIN)),
        "fixed_step_linear_max": int(soft.get("fixed_step_linear_max", FIXED_STEP_LINEAR_MAX)),
        "fixed_step_unit_premium": float(soft.get("fixed_step_unit_premium", FIXED_STEP_UNIT_PREMIUM)),
        "small_account_hard_floor_threshold": float(
            soft.get("small_account_hard_floor_threshold", SMALL_ACCOUNT_HARD_FLOOR_THRESHOLD)
        ),
        "small_account_hard_floor_pct": float(soft.get("small_account_hard_floor_pct", SMALL_ACCOUNT_HARD_FLOOR_PCT)),
    }


def soft_recovery_enabled(
    engine_config: dict[str, Any] | None = None,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> bool:
    """Indica se o Soft Recovery Adaptativo esta ativo (novo bloco ou legado dlambert)."""
    if isinstance(soft_recovery, dict) and ("enabled" in soft_recovery or soft_recovery):
        if "enabled" in soft_recovery:
            return bool(soft_recovery.get("enabled"))
        return True
    cfg = engine_config if isinstance(engine_config, dict) else {}
    nested = cfg.get("soft_recovery")
    if isinstance(nested, dict) and ("enabled" in nested or nested):
        return bool(nested.get("enabled", True))
    return bool(cfg.get("dlambert_enabled", True))


def resolve_amort_cycles(consecutive_losses: int, soft_recovery: dict[str, Any] | None = None) -> int:
    """Fraciona cover de pending em amort_cycles dentro de [min, max]."""
    cfg = soft_recovery if isinstance(soft_recovery, dict) else {}
    amin = max(1, int(cfg.get("amort_cycles_min", DEFAULT_AMORT_CYCLES_MIN)))
    amax = max(amin, int(cfg.get("amort_cycles_max", DEFAULT_AMORT_CYCLES_MAX)))
    span = amax - amin
    losses = max(0, int(consecutive_losses))
    cycles = amax - min(losses, span)
    return max(amin, min(amax, cycles))


def is_recovery_infeasible(
    pending_total: float,
    max_safe_cap: float,
    payout: float,
    soft_recovery: dict[str, Any] | None = None,
) -> bool:
    """True quando pending nao cabe no horizonte amort_cycles_max sob o cap."""
    cfg = soft_recovery if isinstance(soft_recovery, dict) else {}
    amax = max(1, int(cfg.get("amort_cycles_max", DEFAULT_AMORT_CYCLES_MAX)))
    cap = float(max_safe_cap)
    pay = float(payout)
    pending = float(pending_total)
    if pending <= 0.0:
        return False
    if cap <= 0.0 or pay <= 0.0:
        return True
    return (pending / (cap * pay)) > float(amax)


def configured_max_safe_stake_cap(soft_recovery: dict[str, Any] | None) -> float | None:
    """Retorna teto absoluto configurado de soft recovery, se presente."""
    if not isinstance(soft_recovery, dict):
        return None
    raw = soft_recovery.get("max_safe_stake_cap")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def fixed_step_progression_multiplier(
    consecutive_losses: int,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> float | None:
    """Retorna multiplicador U+15% nos niveis lineares 3 e 4; None fora da faixa."""
    cfg = _soft_cfg(soft_recovery)
    lo = int(cfg.get("fixed_step_linear_min", FIXED_STEP_LINEAR_MIN))
    hi = int(cfg.get("fixed_step_linear_max", FIXED_STEP_LINEAR_MAX))
    premium = float(cfg.get("fixed_step_unit_premium", FIXED_STEP_UNIT_PREMIUM))
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
    """Em bancas abaixo de $100, limita stake de recovery a 5% do saldo."""
    cfg = _soft_cfg(soft_recovery)
    threshold = float(cfg.get("small_account_hard_floor_threshold", SMALL_ACCOUNT_HARD_FLOOR_THRESHOLD))
    pct = float(cfg.get("small_account_hard_floor_pct", SMALL_ACCOUNT_HARD_FLOOR_PCT))
    bal = max(0.0, float(bankroll))
    if bal <= 0.0 or bal >= threshold:
        return float(cap)
    return min(float(cap), bal * pct)


def _soft_cfg(soft_recovery: dict[str, Any] | None) -> dict[str, Any]:
    """Normaliza o bloco soft_recovery para dicionario seguro."""
    return soft_recovery if isinstance(soft_recovery, dict) else {}


def is_micro_residual_liability(
    bankroll: float,
    pending_total: float,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> bool:
    """True quando o passivo residual e de baixa intensidade em micro-banca."""
    bal = float(bankroll)
    pending = float(pending_total)
    if bal <= 0.0 or pending <= 0.0:
        return False
    cfg = _soft_cfg(soft_recovery)
    bankroll_max = float(cfg.get("micro_residual_bankroll_max", DEFAULT_MICRO_RESIDUAL_BANKROLL_MAX))
    pending_max = float(cfg.get("micro_residual_pending_max", DEFAULT_MICRO_RESIDUAL_PENDING_MAX))
    pending_pct = float(cfg.get("micro_residual_pending_pct", DEFAULT_MICRO_RESIDUAL_PENDING_PCT))
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
    """Alias semantico de risco de Baixa Intensidade sob Micro Passivo Residual."""
    return is_micro_residual_liability(bankroll, pending_total, soft_recovery=soft_recovery)


def resolve_negative_zscore_veto_floor(
    bankroll: float,
    pending_total: float,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> float:
    """Piso de Z-Score do veto GBDT; relaxa ate -0.60 sob Micro Passivo Residual."""
    cfg = _soft_cfg(soft_recovery)
    if is_micro_residual_liability(bankroll, pending_total, soft_recovery=cfg):
        return float(cfg.get("micro_residual_zscore_floor", DEFAULT_MICRO_RESIDUAL_ZSCORE_FLOOR))
    return DEFAULT_NEGATIVE_ZSCORE_VETO


def resolve_gbdt_waiver_skip_threshold(
    bankroll: float,
    pending_total: float,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> int:
    """Ciclos de inanicao para waiver do GBDT; antecipa sob Micro Passivo Residual."""
    cfg = _soft_cfg(soft_recovery)
    if is_micro_residual_liability(bankroll, pending_total, soft_recovery=cfg):
        return max(1, int(cfg.get("micro_residual_gbdt_waiver_skips", DEFAULT_MICRO_RESIDUAL_GBDT_WAIVER_SKIPS)))
    return max(1, int(cfg.get("gbdt_waiver_skip_cycles", DEFAULT_GBDT_WAIVER_SKIP_CYCLES)))


def cointegration_valve_suppressed(
    bankroll: float,
    pending_total: float,
    *,
    soft_recovery: dict[str, Any] | None = None,
) -> bool:
    """Sob Baixa Intensidade a valvula de cointegracao permanece fechada."""
    return is_low_intensity_recovery(bankroll, pending_total, soft_recovery=soft_recovery)


def risk_session_bankroll_pending(risk_manager: Any | None) -> tuple[float, float, dict[str, Any] | None]:
    """Extrai banca, pending e soft_recovery de um RiskManager opcional."""
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
    """Resolve piso de veto Z-Score a partir do estado vivo de risco."""
    bankroll, pending, soft = risk_session_bankroll_pending(risk_manager)
    if bankroll <= 0.0:
        return DEFAULT_NEGATIVE_ZSCORE_VETO
    return resolve_negative_zscore_veto_floor(bankroll, pending, soft_recovery=soft)


def gbdt_waiver_skip_threshold_for_risk(risk_manager: Any | None) -> int:
    """Resolve limiar de skips do waiver GBDT a partir do estado vivo de risco."""
    bankroll, pending, soft = risk_session_bankroll_pending(risk_manager)
    if bankroll <= 0.0:
        return DEFAULT_GBDT_WAIVER_SKIP_CYCLES
    return resolve_gbdt_waiver_skip_threshold(bankroll, pending, soft_recovery=soft)
