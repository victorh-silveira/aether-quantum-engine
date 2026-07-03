"""Sizing linear D'Alembert: Kelly base + escada aditiva em recovery."""

from __future__ import annotations

from typing import Any

from src.domain.risk.stake_sizing import round_stake


REDIS_DLAMBERT_UNIT_KEY = "session:current:dlambert_unit"
REDIS_DLAMBERT_LINEAR_LOSSES_KEY = "session:current:consecutive_losses_linear"
BOOSTER_DAMPING_FACTOR = 0.50
MAX_LINEAR_LEVEL = 8
MAX_STAKE_U_MULTIPLE = 10.0
MAX_SESSION_DRAWDOWN_U = 25.0


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


def dlambert_circuit_breaker(
    proposed_stake: float,
    *,
    consecutive_losses_linear: int,
    dlambert_unit: float,
    pending_total: float,
    continuous_mode: bool = False,
    stake_min: float = 1.0,
) -> tuple[float, bool]:
    """Trava rigida da escada aditiva: retorna (stake, tripped) contra drawdown superlinear."""
    unit = max(0.0, float(dlambert_unit))
    tripped = (
        int(consecutive_losses_linear) >= MAX_LINEAR_LEVEL
        or (unit > 0.0 and float(proposed_stake) > MAX_STAKE_U_MULTIPLE * unit)
        or (unit > 0.0 and float(pending_total) > MAX_SESSION_DRAWDOWN_U * unit)
    )
    if not tripped:
        return float(proposed_stake), False
    return (float(stake_min) if continuous_mode else 0.0), True


def _continuous_strict_mode(rm: Any) -> bool:
    """Detecta modo continuo estrito para preservar o piso regulamentar minimo."""
    cfg = getattr(rm, "config", None)
    if not isinstance(cfg, dict):
        return False
    execution = cfg.get("orchestrator", {}).get("execution", {})
    return bool(execution.get("mandatory_trade_each_cycle", False))


def _regulatory_stake_min(rm: Any) -> float:
    """Piso regulamentar minimo (stake_min) do gerenciador de risco."""
    params = getattr(rm, "risk_params", None)
    if isinstance(params, dict):
        return float(params.get("stake_min", 1.0))
    return 1.0


def _log_dlambert_circuit_break(
    rm: Any,
    stake: float,
    unit: float,
    consecutive_losses_linear: int,
    pending_total: float,
) -> None:
    """Registra o disparo do circuit breaker aditivo D'Alembert."""
    logger = getattr(rm, "logger", None)
    if logger is None:
        return
    logger.warning(
        "DLAMBERT_CIRCUIT_BREAK | linear=%d | U=$%.2f | pend=$%.2f | stake_travada=$%.2f",
        int(consecutive_losses_linear),
        float(unit),
        float(pending_total),
        float(stake),
    )


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
    """Resolve stake final Kelly ou D'Alembert com circuit breaker de recovery."""
    if recovery_active and dlambert_enabled(dlambert_config):
        unit = resolve_dlambert_unit(kelly_base, rm)
        raw = dlambert_recovery_stake(
            kelly_base,
            unit,
            consecutive_losses_linear,
            pending_total=pending_total,
            bankroll=bankroll,
        )
        guarded, tripped = dlambert_circuit_breaker(
            raw,
            consecutive_losses_linear=consecutive_losses_linear,
            dlambert_unit=unit,
            pending_total=pending_total,
            continuous_mode=_continuous_strict_mode(rm),
            stake_min=_regulatory_stake_min(rm),
        )
        if tripped:
            _log_dlambert_circuit_break(rm, guarded, unit, consecutive_losses_linear, pending_total)
            return round_stake(guarded, recovery_linear=True), "D'ALEMBERT_CB"
        return round_stake(guarded, recovery_linear=True), "D'ALEMBERT"
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
