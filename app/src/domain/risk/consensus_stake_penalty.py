"""Modificador de Kelly por divergencia entre ordem e votos tecnicos."""

from __future__ import annotations

from typing import Any

from src.domain.risk.consensus_recovery_gates import (
    acc_below_recovery_floor,
    adapted_blocks_dal,
    chop_neg_edge_dampens_dal,
    live_evidence_blocks_dal,
    metric_hurst,
)
from src.domain.risk.consensus_stake_helpers import (
    _recovery_waives_consensus_penalty,
    _squeeze_floor_active,
    adaptive_recovery_progression_factor,
    d_squeeze_sovereignty_active,
    neutral_edge_dynamic_unit,
    resolve_contract_payout,
    soft_recovery_progression_multiplier,
    turbo_edge_stake_multiplier,
)
from src.domain.risk.recovery_state_config import load_recovery_state_from_settings
from src.domain.risk.soft_recovery_config import soft_cfg
from src.domain.risk.soft_recovery_explore import (
    apply_forced_explore_early,
    forced_explore_stake,
    soft_floor_scale,
)
from src.domain.risk.soft_recovery_policy import (
    apply_small_account_hard_floor,
    configured_max_safe_stake_cap,
    configured_max_safe_stake_pct,
    fixed_step_progression_multiplier,
    is_recovery_infeasible,
    resolve_amort_cycles,
)
from src.domain.risk.stake_sizing import consensus_entropy_kelly_retention
from src.domain.risk.stake_target_proximity import apply_target_proximity_damping


def resolve_session_base_unit(bankroll: float, base_unit: float, metrics: dict | None) -> float:
    """Resolve unidade base U como max(kelly, neutral*scale_loss_clf) fora do D-SQUEEZE."""
    floor = neutral_edge_dynamic_unit(bankroll) * soft_floor_scale(metrics)
    unit = max(float(base_unit), floor)
    if isinstance(metrics, dict) and not _squeeze_floor_active(metrics):
        metrics["session_base_unit"] = unit
    return unit


def apply_soft_recovery_stake(
    *,
    pending_total: float,
    base_unit: float,
    consecutive_losses: int,
    previous_stake: float,
    bankroll: float,
    metrics: dict | None = None,
    payout: float | None = None,
    risk_params: dict[str, Any] | None = None,
    soft_recovery: dict[str, Any] | None = None,
    session_pnl: float = 0.0,
    target_win: float = 0.0,
) -> float:
    """Aplica progressao adaptativa ou passo fixo quando ha passivo pendente."""
    unit = resolve_session_base_unit(bankroll, base_unit, metrics)
    soft = soft_cfg(soft_recovery)
    material_min = float(soft["material_pending_min"])
    freeze_pct = float(soft["near_stop_win_freeze_pct"])
    target = float(target_win)
    pnl = float(session_pnl)
    near_stop_win = target > 0.0 and (pnl / target) + 1e-12 >= freeze_pct
    pending = float(pending_total)
    material_pending = pending + 1e-12 >= material_min
    cap = max_safe_stake_cap(bankroll, consecutive_losses_linear=consecutive_losses, soft_recovery=soft_recovery)
    hurst_val = metric_hurst(metrics)
    low_hurst_noise = hurst_val is not None and float(hurst_val) < 0.400
    chop_neg_dampen = chop_neg_edge_dampens_dal(metrics)
    acc_force_explore = acc_below_recovery_floor(metrics, consecutive_losses)
    live_force_explore = live_evidence_blocks_dal(metrics, consecutive_losses, soft)
    adapted_force_explore = adapted_blocks_dal(metrics, consecutive_losses, soft)
    if material_pending:
        quality_force_explore = False
        force_early = bool(near_stop_win)
    else:
        quality_force_explore = bool(acc_force_explore or live_force_explore or adapted_force_explore)
        force_early = True
    if force_early:
        return apply_forced_explore_early(
            bankroll=bankroll,
            pending=pending,
            material_pending=material_pending,
            consecutive_losses=consecutive_losses,
            previous_stake=previous_stake,
            unit=unit,
            payout=payout,
            risk_params=risk_params,
            soft=soft,
            soft_recovery=soft_recovery,
            target=target,
            pnl=pnl,
            cap=cap,
            metrics=metrics,
            near_stop_win=near_stop_win,
            low_hurst_noise=low_hurst_noise,
            chop_neg_dampen=chop_neg_dampen,
            acc_force_explore=acc_force_explore,
            live_force_explore=live_force_explore,
            adapted_force_explore=adapted_force_explore,
            quality_force_explore=quality_force_explore,
        )
    factor = adaptive_recovery_progression_factor(payout, risk_params)
    resolved_payout = resolve_contract_payout(payout, risk_params)
    losses = max(0, int(consecutive_losses))
    progression = soft_recovery_progression_multiplier(
        losses, payout=payout, risk_params=risk_params, soft_recovery=soft_recovery
    )
    amort = resolve_amort_cycles(losses, soft_recovery)
    cover_mult = max(1.0, float(soft.get("cover_multiple", 1.0)))
    cover = pending / resolved_payout / float(amort) * cover_mult
    horizon_infeasible = is_recovery_infeasible(pending, cap, resolved_payout, soft_recovery)
    cover_blocked = cover + 1e-12 >= cap
    infeasible = bool(horizon_infeasible or cover_blocked)
    force_explore = bool(soft["infeasible_force_explore"]) and infeasible
    if force_explore:
        explore = forced_explore_stake(
            bankroll=bankroll,
            pending=pending,
            material_pending=False,
            consecutive_losses=losses,
            payout=payout,
            risk_params=risk_params,
            soft=soft,
            target=target,
            pnl=pnl,
            cap=cap,
            metrics=metrics,
            reason="infeasible",
        )
        if isinstance(metrics, dict):
            metrics["recovery_soft_progression"] = factor
            metrics["recovery_adaptive_payout"] = resolved_payout
            metrics["recovery_soft_losses"] = losses
            metrics["recovery_soft_anchor_stake"] = float(previous_stake) if float(previous_stake) > 0.0 else unit
            metrics["recovery_cover_need"] = cover
            metrics["recovery_cover_multiple"] = cover_mult
            metrics["recovery_amort_cycles"] = amort
            metrics["recovery_fixed_step"] = (
                fixed_step_progression_multiplier(losses, soft_recovery=soft_recovery) is not None
            )
            metrics["recovery_progression_multiplier"] = float(progression)
            metrics["recovery_infeasible"] = True
            metrics["recovery_force_explore"] = True
            metrics["recovery_material_pending"] = True
            metrics["recovery_near_stop_win_freeze"] = False
        return explore
    stake = float(cover)
    if int(amort) <= 1:
        progression = 1.0
    elif target > 0.0:
        stake = apply_target_proximity_damping(stake, target, pnl)
    if isinstance(metrics, dict):
        metrics["recovery_soft_progression"] = factor
        metrics["recovery_adaptive_payout"] = resolved_payout
        metrics["recovery_soft_losses"] = losses
        metrics["recovery_soft_anchor_stake"] = float(previous_stake) if float(previous_stake) > 0.0 else unit
        metrics["recovery_cover_need"] = cover
        metrics["recovery_cover_multiple"] = cover_mult
        metrics["recovery_amort_cycles"] = amort
        metrics["recovery_fixed_step"] = (
            fixed_step_progression_multiplier(losses, soft_recovery=soft_recovery) is not None
        )
        metrics["recovery_progression_multiplier"] = float(progression)
        metrics["recovery_infeasible"] = bool(infeasible)
        metrics["recovery_force_explore"] = False
        metrics["recovery_material_pending"] = True
        metrics["recovery_near_stop_win_freeze"] = False
        metrics["recovery_acc_force_explore"] = False
        metrics["recovery_live_force_explore"] = False
        metrics["recovery_adapted_force_explore"] = False
    return min(stake, cap)


def max_safe_stake_cap(
    bankroll: float, *, consecutive_losses_linear: int = 0, soft_recovery: dict[str, Any] | None = None
) -> float:
    """Retorna teto absoluto; micro-banca <$100 limita recovery a 5% do saldo."""
    linear = max(0, int(consecutive_losses_linear))
    bal = max(0.0, float(bankroll))
    rs = load_recovery_state_from_settings()
    if bal <= float(rs["micro_bankroll_threshold"]) and linear >= int(rs["micro_tail_linear_level"]):
        configured = configured_max_safe_stake_cap(soft_recovery)
        raw = (
            configured
            if configured is not None
            else float(rs["micro_tail_unit_multiplier"]) * neutral_edge_dynamic_unit(bal)
        )
        return apply_small_account_hard_floor(raw, bal, soft_recovery=soft_recovery)
    pct = configured_max_safe_stake_pct(soft_recovery)
    soft = soft_cfg(soft_recovery)
    if linear >= 3:
        pct = min(pct, float(soft["max_safe_stake_pct_linear3"]))
    elif linear >= 2:
        pct = min(pct, float(soft["max_safe_stake_pct_linear2"]))
    return apply_small_account_hard_floor(bal * pct, bal, soft_recovery=soft_recovery)


def enforce_d_squeeze_stake_floor(
    final_stake: float, stake_min: float, metrics: dict | None, *, pending_total: float = 0.0
) -> float:
    """Comprime stake ao piso absoluto da API quando D-SQUEEZE revoga recovery."""
    if not d_squeeze_sovereignty_active(metrics):
        return final_stake
    if float(pending_total) > 0.0:
        if isinstance(metrics, dict):
            metrics["d_squeeze_floor_waived_for_recovery"] = True
        return final_stake
    if isinstance(metrics, dict):
        metrics["d_squeeze_recovery_waiver_revoked"] = True
    return float(stake_min)


def apply_neutral_edge_kelly_base(kelly_base: float, bankroll: float, metrics: dict | None) -> float:
    """Eleva kelly_base ao piso neutral; so loss_clf soft escala o piso."""
    if isinstance(metrics, dict) and _squeeze_floor_active(metrics):
        return kelly_base
    floor = neutral_edge_dynamic_unit(bankroll) * soft_floor_scale(metrics)
    base = max(float(kelly_base), floor)
    if isinstance(metrics, dict) and not _squeeze_floor_active(metrics):
        metrics["session_base_unit"] = base
    return base


def apply_turbo_edge_stake(final_stake: float, metrics: dict | None) -> float:
    """Aplica multiplicador turbo sobre stake final quando conviccao de cauda e extrema."""
    mult = turbo_edge_stake_multiplier(metrics)
    if mult > 1.0 and isinstance(metrics, dict):
        metrics["consensus_turbo_edge_active"] = True
    return float(final_stake) * mult


def consensus_kelly_retention(
    metrics: dict,
    order_direction: str | None,
    *,
    kelly_config: dict[str, Any] | None = None,
    consecutive_losses: int = 0,
    pending_loss_total: float = 0.0,
) -> float:
    """Retorna fator [floor, 1.0] para atenuar f* quando ord diverge do consenso tecnico."""
    if isinstance(metrics, dict) and _squeeze_floor_active(metrics):
        metrics["consensus_penalty_d_squeeze"] = True
        cfg = kelly_config if isinstance(kelly_config, dict) else {}
        return float(cfg.get("consensus_min_retention", 1.0 - float(cfg.get("consensus_max_cut", 0.50))))
    if isinstance(metrics, dict):
        recovering = float(pending_loss_total) > 0.0 or int(consecutive_losses) > 0
        if recovering and _recovery_waives_consensus_penalty(
            metrics,
            kelly_config or {},
            consecutive_losses=int(consecutive_losses),
            pending_loss_total=float(pending_loss_total),
            order_direction=order_direction,
        ):
            metrics["consensus_penalty_recovery_waived"] = True
            return 1.0
    return consensus_entropy_kelly_retention(metrics, order_direction, kelly_config=kelly_config)


def cross_veto_recovery_waiver_allowed(
    metrics: dict[str, Any] | None, *, direction: str | None, risk_manager: Any | None = None
) -> bool:
    """Verifica se o waiver de recovery para o veto cruzado esta ativo e permitido."""
    if metrics is None or direction is None:
        return False
    from src.domain.risk.risk_recovery_state import meta_payoff_veto_emergency_waiver  # noqa: PLC0415

    return meta_payoff_veto_emergency_waiver(metrics, direction=direction, risk_manager=risk_manager)
