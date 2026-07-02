"""Sizing linear D'Alembert: Kelly base + escada aditiva em recovery."""

from __future__ import annotations

from typing import Any

from src.domain.risk.stake_sizing import round_stake


REDIS_DLAMBERT_UNIT_KEY = "session:current:dlambert_unit"
REDIS_DLAMBERT_LINEAR_LOSSES_KEY = "session:current:consecutive_losses_linear"
BOOSTER_DAMPING_FACTOR = 0.50


def dlambert_enabled(dlambert_config: dict[str, Any]) -> bool:
    """Indica se o motor D'Alembert esta ativo."""
    return bool(dlambert_config.get("dlambert_enabled", True))


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


def dlambert_amortization_multiplier(
    pending_total: float,
    bankroll: float,
    *,
    damping: float = BOOSTER_DAMPING_FACTOR,
) -> float:
    """Fator de aceleracao amortecida: 1 + min(1.5, pend/(banca*0.02)) * damping."""
    if pending_total <= 0.0 or bankroll <= 0.0:
        return 1.0
    ratio = min(1.5, float(pending_total) / (float(bankroll) * 0.02))
    return 1.0 + ratio * max(0.0, float(damping))


def effective_dlambert_unit(
    unit: float,
    pending_total: float,
    bankroll: float,
    *,
    damping: float = BOOSTER_DAMPING_FACTOR,
) -> float:
    """Unidade linear efetiva com expansao maxima de 1.75x quando damping=0.50."""
    u = max(0.0, float(unit))
    pending = float(pending_total)
    br = float(bankroll)
    if pending <= 0.0 or br <= 0.0:
        return u
    return u * dlambert_amortization_multiplier(pending, br, damping=damping)


def dlambert_recovery_stake(
    kelly_base: float,
    unit: float,
    consecutive_losses_linear: int,
    *,
    pending_total: float = 0.0,
    bankroll: float = 0.0,
) -> float:
    """Calcula stake linear: Kelly atual + perdas lineares * U efetivo."""
    linear = max(0, int(consecutive_losses_linear))
    base = max(0.0, float(kelly_base))
    u_eff = effective_dlambert_unit(unit, pending_total, bankroll)
    return base + linear * u_eff


def resolve_dlambert_stake(
    *,
    recovery_active: bool,
    bankroll: float,
    kelly_base: float,
    dlambert_config: dict[str, Any],
    rm: Any,
    consecutive_losses_linear: int,
    pending_total: float = 0.0,
) -> tuple[float, str]:
    """Resolve stake final Kelly ou D'Alembert com arredondamento."""
    if recovery_active and dlambert_enabled(dlambert_config):
        unit = resolve_dlambert_unit(kelly_base, rm)
        raw = dlambert_recovery_stake(
            kelly_base,
            unit,
            consecutive_losses_linear,
            pending_total=pending_total,
            bankroll=bankroll,
        )
        return round_stake(raw, recovery_linear=True), "D'ALEMBERT"
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
) -> str:
    """Monta sufixo de log com detalhes da stake D'Alembert."""
    _ = dlambert_config
    if mode_tag != "D'ALEMBERT":
        return ""
    unit = float(dlambert_unit)
    u_eff = effective_dlambert_unit(unit, loss_to_recover, bankroll)
    linear = int(consecutive_losses_linear)
    return (
        f" | D'ALEMBERT ${final_stake:.2f} (kelly=${kelly_base:.2f}+"
        f"{linear}*U_eff=${u_eff:.2f}) | pend=${loss_to_recover:.2f}"
    )
