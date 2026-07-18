"""Helpers de fracao Kelly e caps de stake para risk_stake_calc."""

from typing import Any

from src.domain.risk.consensus_stake_penalty import max_safe_stake_cap
from src.domain.risk.kelly_f_star_adjustments import (
    apply_consensus_entropy_f_star,
    apply_kelly_fraction_scale,
    kelly_base_with_consensus_floor,
)
from src.domain.risk.stake_sizing import clamp_kelly_stake
from src.domain.risk.super_concordance_kelly import apply_super_concordance_kelly_fraction


def resolve_f_star_and_kelly_base(
    rm: Any,
    *,
    bankroll: float,
    kelly_f: float,
    sizing_conviction: float,
    dl_metrics: dict | None,
    order_direction: Any,
    recovery_active: bool,
    recovery_bypass_consensus: bool,
    silent: bool,
) -> tuple[float, float]:
    """Resolve f* e kelly_base com consensus/scale/floor."""
    f_star = apply_super_concordance_kelly_fraction(
        kelly_f,
        rm.kelly_config,
        dl_metrics if isinstance(dl_metrics, dict) else None,
        order_direction,
        recovery_active=recovery_active,
    )
    if not recovery_bypass_consensus:
        f_star = apply_consensus_entropy_f_star(rm, f_star, dl_metrics, order_direction, silent=silent)
    elif isinstance(dl_metrics, dict):
        dl_metrics["consensus_entropy_retention"] = 1.0
    f_star = apply_kelly_fraction_scale(f_star, dl_metrics)
    stake_min = float(rm.risk_params.get("stake_min", 1.0))
    kelly_base = (
        clamp_kelly_stake(bankroll, bankroll * f_star, rm.kelly_config, sizing_conviction)
        if recovery_bypass_consensus
        else kelly_base_with_consensus_floor(
            bankroll, f_star, dl_metrics, rm.kelly_config, sizing_conviction, stake_min
        )
    )
    return f_star, kelly_base


def cap_final_stake(
    final_stake: float,
    *,
    bankroll: float,
    conviction: float,
    recovery_stress: bool,
    linear_losses: int,
    rm: Any,
) -> tuple[float, float]:
    """Aplica safe_cap e max_stake_pct; retorna (stake, safe_cap)."""
    soft_cfg = getattr(rm, "soft_recovery_config", None)
    soft = soft_cfg if isinstance(soft_cfg, dict) else None
    safe_cap = max_safe_stake_cap(bankroll, consecutive_losses_linear=linear_losses, soft_recovery=soft)
    max_pct = float(rm.kelly_config.get("max_stake_pct", 0.035) or 0.035)
    hi_thr = float(rm.kelly_config.get("high_conviction_stake_threshold", 1.01))
    if float(conviction) + 1e-9 >= hi_thr:
        max_pct = float(rm.kelly_config.get("max_stake_pct_high_conviction", max_pct) or max_pct)
    pct_cap = max(0.0, float(bankroll) * max_pct)
    capped = min(final_stake, safe_cap) if recovery_stress else min(final_stake, safe_cap, pct_cap)
    return capped, safe_cap
