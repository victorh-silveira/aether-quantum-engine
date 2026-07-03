"""Sizing Kelly base acoplado a Martingale Geometrico puro sem teto em recovery."""

from __future__ import annotations

from typing import Any

from src.domain.risk.stake_sizing import round_stake


REDIS_DLAMBERT_UNIT_KEY = "session:current:dlambert_unit"
REDIS_DLAMBERT_LINEAR_LOSSES_KEY = "session:current:consecutive_losses_linear"
GEOMETRIC_MARTINGALE_BASE = 2.0


def dlambert_enabled(dlambert_config: dict[str, Any]) -> bool:
    """Indica se o motor de recovery Martingale esta ativo."""
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


def geometric_martingale_stake(
    kelly_base: float,
    consecutive_losses_linear: int,
) -> float:
    """Curva multiplicativa classica: Kelly_Base * 2^perdas_consecutivas, sem teto."""
    base = max(0.0, float(kelly_base))
    losses = max(0, int(consecutive_losses_linear))
    return base * (GEOMETRIC_MARTINGALE_BASE**losses)


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
    """Resolve stake final Kelly ou Martingale Geometrico contínuo em recovery."""
    _ = (bankroll, pending_total)
    if recovery_active and dlambert_enabled(dlambert_config):
        resolve_dlambert_unit(kelly_base, rm)
        raw = geometric_martingale_stake(kelly_base, consecutive_losses_linear)
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
    """Monta sufixo de log com detalhes da stake Martingale Geometrico."""
    _ = (dlambert_unit, dlambert_config, bankroll)
    if mode_tag != "D'ALEMBERT":
        return ""
    linear = int(consecutive_losses_linear)
    return f" | D'ALEMBERT ${final_stake:.2f} (kelly=${kelly_base:.2f}*2^{linear}) | pend=${loss_to_recover:.2f}"
