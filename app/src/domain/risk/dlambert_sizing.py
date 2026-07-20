"""Sizing Kelly base acoplado a progressao adaptativa indexada ao payout em recovery."""

from __future__ import annotations

from typing import Any

from src.domain.risk.consensus_stake_penalty import (
    adaptive_recovery_progression_factor,
    apply_soft_recovery_stake,
    max_safe_stake_cap,
    resolve_contract_payout,
    resolve_session_base_unit,
    soft_recovery_progression_multiplier,
)
from src.domain.risk.soft_recovery_policy import soft_recovery_enabled
from src.domain.risk.stake_sizing import round_stake


REDIS_DLAMBERT_UNIT_KEY = "session:current:dlambert_unit"
REDIS_DLAMBERT_LINEAR_LOSSES_KEY = "session:current:consecutive_losses_linear"


def dlambert_enabled(dlambert_config: dict[str, Any], *, soft_recovery: dict[str, Any] | None = None) -> bool:
    """Indica se o Soft Recovery Adaptativo esta ativo (alias legado)."""
    return soft_recovery_enabled(dlambert_config, soft_recovery=soft_recovery)


def _soft_cfg(rm: Any, dlambert_config: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve bloco soft_recovery a partir do RiskManager ou config legada."""
    soft = getattr(rm, "soft_recovery_config", None)
    if isinstance(soft, dict) and soft:
        return soft
    nested = dlambert_config.get("soft_recovery")
    return nested if isinstance(nested, dict) and nested else None


def resolve_dlambert_unit(
    kelly_base: float,
    rm: Any,
) -> float:
    """Resolve unidade base U: override de config ou primeira stake Kelly da sessao."""
    cfg = getattr(rm, "dlambert_config", {}) or {}
    override = cfg.get("dlambert_unit_override")
    if override is not None:
        try:
            unit = float(override)
        except (TypeError, ValueError):
            unit = 0.0
        if unit > 0.0:
            rm.dlambert_unit = unit
            return unit
    existing = float(getattr(rm, "dlambert_unit", 0.0))
    if existing > 0.0:
        return existing
    if kelly_base > 0.0:
        rm.dlambert_unit = float(kelly_base)
        return float(kelly_base)
    return 0.0


def _resolve_override_value(dlambert_config: dict[str, Any]) -> float:
    """Extrai override positivo de unidade U a partir da config dlambert."""
    override = dlambert_config.get("dlambert_unit_override")
    if override is None:
        return 0.0
    try:
        value = float(override)
    except (TypeError, ValueError):
        return 0.0
    return value if value > 0.0 else 0.0


def effective_martingale_base(
    kelly_base: float,
    rm: Any,
    dlambert_config: dict[str, Any],
) -> float:
    """Resolve base ancorada U com piso de override para progressao suave."""
    existing = float(getattr(rm, "dlambert_unit", 0.0))
    unit = existing if existing > 0.0 else resolve_dlambert_unit(kelly_base, rm)
    override = _resolve_override_value(dlambert_config)
    return max(override, unit, 0.0)


def martingale_recovery_active(
    *,
    recovery_active: bool,
    pending_total: float,
    consecutive_losses_linear: int,
) -> bool:
    """Indica estresse de recovery quando ha passivo pendente ou perdas lineares."""
    return recovery_active or float(pending_total) > 0.0 or int(consecutive_losses_linear) > 0


def resolve_dlambert_stake(
    *,
    recovery_active: bool,
    bankroll: float,
    kelly_base: float,
    dlambert_config: dict[str, Any],
    rm: Any,
    consecutive_losses_linear: int,
    pending_total: float = 0.0,
    payout: float | None = None,
    dl_metrics: dict | None = None,
) -> tuple[float, str]:
    """Resolve stake final Kelly ou Soft Recovery Adaptativo indexado ao payout."""
    soft = _soft_cfg(rm, dlambert_config)
    stress_recovery = martingale_recovery_active(
        recovery_active=recovery_active,
        pending_total=pending_total,
        consecutive_losses_linear=consecutive_losses_linear,
    )
    if stress_recovery and soft_recovery_enabled(dlambert_config, soft_recovery=soft):
        effective_base = effective_martingale_base(kelly_base, rm, dlambert_config)
        metrics = dl_metrics if isinstance(dl_metrics, dict) else None
        session_base = resolve_session_base_unit(bankroll, effective_base, metrics)
        previous_stake = float(getattr(rm, "last_loss_stake", 0.0))
        raw = apply_soft_recovery_stake(
            pending_total=float(pending_total),
            base_unit=session_base,
            consecutive_losses=int(consecutive_losses_linear),
            previous_stake=previous_stake,
            bankroll=bankroll,
            metrics=metrics,
            payout=payout,
            risk_params=getattr(rm, "risk_params", None),
            soft_recovery=soft,
            session_pnl=float(getattr(rm, "total_session_profit", 0.0) or 0.0),
            target_win=float(getattr(rm, "daily_stop_win_target", 0.0) or 0.0),
        )
        rounded = round_stake(raw, recovery_linear=True)
        cap = max_safe_stake_cap(
            bankroll,
            consecutive_losses_linear=consecutive_losses_linear,
            soft_recovery=soft,
        )
        return min(rounded, cap), "D'ALEMBERT"
    resolve_dlambert_unit(kelly_base, rm)
    return round_stake(float(kelly_base), recovery_linear=False), "KELLY"


def dlambert_log_suffix(
    mode_tag: str,
    final_stake: float,
    loss_to_recover: float,
    kelly_base: float,
    *,
    dlambert_unit: float = 0.0,
    consecutive_losses_linear: int = 0,
    dlambert_config: dict[str, Any] | None = None,
    bankroll: float = 0.0,
    payout: float | None = None,
) -> str:
    """Monta sufixo de log com detalhes da progressao adaptativa ou passo fixo."""
    _ = (dlambert_unit, dlambert_config, bankroll)
    if mode_tag != "D'ALEMBERT":
        return ""
    linear = int(consecutive_losses_linear)
    resolved_payout = resolve_contract_payout(payout)
    factor = adaptive_recovery_progression_factor(payout)
    mult = soft_recovery_progression_multiplier(linear, payout=payout)
    if 3 <= linear <= 4:
        return (
            f" | D'ALEMBERT ${final_stake:.2f} (fixed=U+15% x{mult:.2f} n={linear}"
            f" p={resolved_payout:.2f} U=${kelly_base:.2f}) | pend=${loss_to_recover:.2f}"
        )
    return (
        f" | D'ALEMBERT ${final_stake:.2f} (soft={factor:.2f}x^{linear}"
        f" p={resolved_payout:.2f} U=${kelly_base:.2f}) | pend=${loss_to_recover:.2f}"
    )
