"""Sizing SIDE_EQ soft: atenua Kelly no lado toxico sem SKIP/veto de direcao."""

from __future__ import annotations

from typing import Any

from src.application.services.side_equilibrium_store import side_eq_config_from_orch, snapshot_side_counts
from src.domain.analytics.side_equilibrium import (
    ACTION_HARD_SKIP,
    ACTION_PASS,
    ACTION_SOFT,
    SideEquilibriumDecision,
    evaluate_side_equilibrium,
)
from src.domain.models.trade import TradeDirection


def _soft_from_decision(decision: SideEquilibriumDecision, kelly_mult_soft: float) -> SideEquilibriumDecision:
    """Converte hard_skip de dominio em soft_penalty de sizing."""
    if decision.action != ACTION_HARD_SKIP:
        return decision
    return SideEquilibriumDecision(
        action=ACTION_SOFT,
        reason=f"sizing_{decision.reason}",
        kelly_mult=float(kelly_mult_soft),
        margin_boost=0.0,
        call_n=decision.call_n,
        call_wins=decision.call_wins,
        put_n=decision.put_n,
        put_wins=decision.put_wins,
        side_wr=decision.side_wr,
        freq_bias=decision.freq_bias,
        z_vs_half=decision.z_vs_half,
    )


def _pick_sizing_decision(
    *,
    small: SideEquilibriumDecision,
    large: SideEquilibriumDecision,
    kelly_mult_soft: float,
) -> SideEquilibriumDecision:
    """Prioriza soft large-N; mapeia hard small-N para soft sizing."""
    large_soft = _soft_from_decision(large, kelly_mult_soft)
    small_soft = _soft_from_decision(small, kelly_mult_soft)
    if large_soft.action == ACTION_SOFT:
        return large_soft
    if small_soft.action == ACTION_SOFT:
        return small_soft
    return large_soft if large_soft.reason != "disabled" else small_soft


def apply_side_eq_kelly_sizing(
    orch: Any | None,
    symbol: str | None,
    direction: TradeDirection,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Aplica kelly_fraction_scale no lado toxico; nunca bloqueia direcao."""
    metrics.pop("side_eq_blocked", None)
    metrics["side_eq_blocked"] = False
    if orch is None or not symbol:
        metrics["side_eq_action"] = ACTION_PASS
        metrics["side_eq_reason"] = "no_orch"
        return metrics
    cfg = side_eq_config_from_orch(orch)
    if not cfg.enabled:
        metrics["side_eq_action"] = ACTION_PASS
        metrics["side_eq_reason"] = "disabled"
        return metrics
    proposed = direction.name if isinstance(direction, TradeDirection) else str(direction).upper()
    small_counts = snapshot_side_counts(orch, str(symbol), window=cfg.small_window)
    large_counts = snapshot_side_counts(orch, str(symbol), window=cfg.large_window)
    small = evaluate_side_equilibrium(small_counts, proposed, config=cfg, regime="small")
    large = evaluate_side_equilibrium(large_counts, proposed, config=cfg, regime="large")
    decision = _pick_sizing_decision(small=small, large=large, kelly_mult_soft=cfg.kelly_mult_soft)
    metrics["side_eq_action"] = decision.action if decision.action != ACTION_HARD_SKIP else ACTION_SOFT
    metrics["side_eq_reason"] = decision.reason
    metrics["side_eq_side_wr"] = decision.side_wr
    metrics["side_eq_freq_bias"] = decision.freq_bias
    metrics["side_call_n"] = decision.call_n
    metrics["side_put_n"] = decision.put_n
    metrics["side_eq_z"] = decision.z_vs_half
    if decision.action == ACTION_SOFT:
        scale = float(metrics.get("kelly_fraction_scale", 1.0))
        metrics["kelly_fraction_scale"] = max(0.05, scale * float(decision.kelly_mult))
        metrics["side_eq_kelly_mult"] = float(decision.kelly_mult)
    else:
        metrics["side_eq_kelly_mult"] = 1.0
    return metrics
