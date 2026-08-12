"""EXPLORE forçado sob soft-signal / freeze: cover ou piso, nunca U sticky."""

from __future__ import annotations

from typing import Any

from src.domain.risk.consensus_stake_helpers import (
    neutral_edge_dynamic_unit,
    resolve_contract_payout,
)
from src.domain.risk.soft_recovery_policy import resolve_amort_cycles
from src.domain.risk.stake_target_proximity import apply_target_proximity_damping


def soft_floor_scale(metrics: dict | None) -> float:
    """Piso neutral permanece 100% da banca SSOT (loss_clf soft nao esmaga U)."""
    _ = metrics
    return 1.0


def neutral_explore_floor(bankroll: float, metrics: dict | None) -> float:
    """Piso EXPLORE forçado: neutral_bankroll_pct, sem U sticky da sessao."""
    return neutral_edge_dynamic_unit(bankroll) * soft_floor_scale(metrics)


def damped_cover_stake(
    *,
    pending: float,
    consecutive_losses: int,
    payout: float | None,
    risk_params: dict[str, Any] | None,
    soft: dict[str, Any],
    target: float,
    pnl: float,
) -> tuple[float, float, int]:
    """Cover amortizado com damping de meta; retorna (stake_bruto, cover, amort)."""
    resolved_payout = resolve_contract_payout(payout, risk_params)
    amort = resolve_amort_cycles(consecutive_losses, soft)
    cover_mult = max(1.0, float(soft.get("cover_multiple", 1.0)))
    cover = float(pending) / resolved_payout / float(amort) * cover_mult
    stake = float(cover)
    if int(amort) > 1 and target > 0.0:
        stake = apply_target_proximity_damping(stake, target, pnl)
    return stake, cover, int(amort)


def forced_explore_stake(
    *,
    bankroll: float,
    pending: float,
    material_pending: bool,
    consecutive_losses: int,
    payout: float | None,
    risk_params: dict[str, Any] | None,
    soft: dict[str, Any],
    target: float,
    pnl: float,
    cap: float,
    metrics: dict | None,
    reason: str,
) -> float:
    """EXPLORE forçado: cover∩piso se ha pending material; senao so piso neutral."""
    floor = neutral_explore_floor(bankroll, metrics)
    used_cover = False
    cover_need = 0.0
    amort = 0
    if material_pending and pending > 0.0 and reason != "infeasible":
        damped, cover_need, amort = damped_cover_stake(
            pending=pending,
            consecutive_losses=consecutive_losses,
            payout=payout,
            risk_params=risk_params,
            soft=soft,
            target=target,
            pnl=pnl,
        )
        stake = min(damped, floor, cap)
        used_cover = True
    else:
        stake = min(floor, cap)
    if isinstance(metrics, dict):
        metrics["recovery_explore_used_cover"] = used_cover
        metrics["recovery_force_explore_reason"] = reason
        metrics["recovery_explore_neutral_floor"] = round(float(floor), 6)
        metrics["recovery_cover_need"] = float(cover_need)
        if amort > 0:
            metrics["recovery_amort_cycles"] = amort
            metrics["recovery_cover_multiple"] = max(1.0, float(soft.get("cover_multiple", 1.0)))
    return max(0.0, float(stake))
