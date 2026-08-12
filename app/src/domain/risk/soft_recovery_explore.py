"""EXPLORE forçado sob soft-signal / freeze: piso neutral, nunca cover nem U sticky."""

from __future__ import annotations

from typing import Any

from src.domain.risk.consensus_stake_helpers import (
    neutral_edge_dynamic_unit,
    resolve_contract_payout,
)
from src.domain.risk.soft_recovery_policy import is_recovery_infeasible, resolve_amort_cycles
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


def soft_early_infeasible(
    *,
    pending: float,
    material_pending: bool,
    consecutive_losses: int,
    payout: float | None,
    risk_params: dict[str, Any] | None,
    soft: dict[str, Any],
    soft_recovery: dict[str, Any] | None,
    cap: float,
) -> bool:
    """True quando soft early-return teria cover inviavel (telemetria INFEASIBLE)."""
    if not material_pending or pending <= 0.0:
        return False
    resolved_payout = resolve_contract_payout(payout, risk_params)
    amort = resolve_amort_cycles(max(0, int(consecutive_losses)), soft_recovery)
    cover_mult = max(1.0, float(soft.get("cover_multiple", 1.0)))
    cover = pending / resolved_payout / float(amort) * cover_mult
    return bool(is_recovery_infeasible(pending, cap, resolved_payout, soft_recovery) or cover + 1e-12 >= cap)


def mark_forced_explore_metrics(
    metrics: dict | None,
    *,
    consecutive_losses: int,
    previous_stake: float,
    unit: float,
    material_pending: bool,
    near_stop_win: bool,
    low_hurst_noise: bool,
    chop_neg_dampen: bool,
    acc_force_explore: bool,
    live_force_explore: bool,
    adapted_force_explore: bool,
    quality_force_explore: bool,
    soft_infeasible: bool,
) -> None:
    """Preenche telemetria do early-return EXPLORE forçado."""
    if not isinstance(metrics, dict):
        return
    metrics["recovery_soft_progression"] = 1.0
    metrics["recovery_soft_losses"] = max(0, int(consecutive_losses))
    metrics["recovery_soft_anchor_stake"] = float(previous_stake) if float(previous_stake) > 0.0 else unit
    metrics["recovery_material_pending"] = bool(material_pending)
    metrics["recovery_near_stop_win_freeze"] = bool(near_stop_win)
    metrics["recovery_low_hurst_damped"] = bool(low_hurst_noise)
    metrics["recovery_chop_neg_edge_damped"] = bool(chop_neg_dampen)
    metrics["recovery_acc_force_explore"] = bool(acc_force_explore and not material_pending)
    metrics["recovery_live_force_explore"] = bool(live_force_explore and not material_pending)
    metrics["recovery_adapted_force_explore"] = bool(adapted_force_explore and not material_pending)
    metrics["recovery_progression_multiplier"] = 1.0
    metrics["recovery_infeasible"] = bool(soft_infeasible)
    metrics["recovery_force_explore"] = bool(
        quality_force_explore or low_hurst_noise or chop_neg_dampen or near_stop_win
    )


def force_early_explore_reason(
    *,
    near_stop_win: bool,
    low_hurst_noise: bool,
    chop_neg_dampen: bool,
    quality_force_explore: bool,
) -> str:
    """Motivo telemetrico do EXPLORE forçado no early-return."""
    if near_stop_win:
        return "near_stop"
    if low_hurst_noise:
        return "low_hurst"
    if chop_neg_dampen:
        return "neg_edge"
    if quality_force_explore:
        return "quality"
    return "no_material_pending"


def apply_forced_explore_early(
    *,
    bankroll: float,
    pending: float,
    material_pending: bool,
    consecutive_losses: int,
    previous_stake: float,
    unit: float,
    payout: float | None,
    risk_params: dict[str, Any] | None,
    soft: dict[str, Any],
    soft_recovery: dict[str, Any] | None,
    target: float,
    pnl: float,
    cap: float,
    metrics: dict | None,
    near_stop_win: bool,
    low_hurst_noise: bool,
    chop_neg_dampen: bool,
    acc_force_explore: bool,
    live_force_explore: bool,
    adapted_force_explore: bool,
    quality_force_explore: bool,
) -> float:
    """EXPLORE early-return: piso + telemetria (incl. INFEASIBLE se cover inviavel)."""
    reason = force_early_explore_reason(
        near_stop_win=near_stop_win,
        low_hurst_noise=low_hurst_noise,
        chop_neg_dampen=chop_neg_dampen,
        quality_force_explore=quality_force_explore,
    )
    explore = forced_explore_stake(
        bankroll=bankroll,
        pending=pending,
        material_pending=bool(material_pending and pending > 0.0),
        consecutive_losses=consecutive_losses,
        payout=payout,
        risk_params=risk_params,
        soft=soft,
        target=target,
        pnl=pnl,
        cap=cap,
        metrics=metrics,
        reason=reason,
    )
    soft_infeasible = soft_early_infeasible(
        pending=pending,
        material_pending=material_pending,
        consecutive_losses=consecutive_losses,
        payout=payout,
        risk_params=risk_params,
        soft=soft,
        soft_recovery=soft_recovery,
        cap=cap,
    )
    mark_forced_explore_metrics(
        metrics,
        consecutive_losses=consecutive_losses,
        previous_stake=previous_stake,
        unit=unit,
        material_pending=material_pending,
        near_stop_win=near_stop_win,
        low_hurst_noise=low_hurst_noise,
        chop_neg_dampen=chop_neg_dampen,
        acc_force_explore=acc_force_explore,
        live_force_explore=live_force_explore,
        adapted_force_explore=adapted_force_explore,
        quality_force_explore=quality_force_explore,
        soft_infeasible=soft_infeasible,
    )
    return explore


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
    """EXPLORE forçado: sempre piso neutral; cover so no caminho DAL."""
    _ = (pending, material_pending, consecutive_losses, payout, risk_params, soft, target, pnl)
    floor = neutral_explore_floor(bankroll, metrics)
    stake = min(max(0.0, floor), cap)
    if isinstance(metrics, dict):
        metrics["recovery_explore_used_cover"] = False
        metrics["recovery_force_explore_reason"] = reason
        metrics["recovery_explore_neutral_floor"] = round(float(floor), 6)
        metrics["recovery_cover_need"] = 0.0
    return max(0.0, float(stake))
