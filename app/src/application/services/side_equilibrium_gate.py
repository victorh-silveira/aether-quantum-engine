"""Gate de equilibrio CALL/PUT: small-N hard skip, large-N soft Kelly/margem."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.side_equilibrium_helpers import (
    alternate_side_is_preferable,
    flip_conflicts_price_zone,
    log_side_eq_flip,
    positive_meta_edge_keeps_proposed,
    primary_side_is_toxic,
    soft_keep_proposed,
    thin_margin_blocks_flip,
)
from src.application.services.side_equilibrium_store import side_eq_config_from_orch, snapshot_side_counts
from src.domain.analytics.side_equilibrium import (
    ACTION_HARD_SKIP,
    ACTION_PASS,
    ACTION_SOFT,
    SideEquilibriumDecision,
    evaluate_side_equilibrium,
)
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")


def opposite_trade_direction(side: TradeDirection) -> TradeDirection:
    """Retorna o lado oposto CALL/PUT."""
    return TradeDirection.CALL if side == TradeDirection.PUT else TradeDirection.PUT


def evaluate_proposed_side_equilibrium(
    orch: Any | None, symbol: str | None, proposed: TradeDirection
) -> SideEquilibriumDecision:
    """Avalia equilibrio CALL/PUT para o lado proposto."""
    if orch is None or not symbol:
        return SideEquilibriumDecision(action=ACTION_PASS, reason="no_context")
    cfg = side_eq_config_from_orch(orch)
    if not cfg.enabled:
        return SideEquilibriumDecision(action=ACTION_PASS, reason="disabled")
    small = snapshot_side_counts(orch, str(symbol), window=cfg.small_window)
    large = snapshot_side_counts(orch, str(symbol), window=cfg.large_window)
    small_decision = evaluate_side_equilibrium(small, proposed.name, config=cfg, regime="small")
    if small_decision.action == ACTION_HARD_SKIP:
        return small_decision
    return evaluate_side_equilibrium(large, proposed.name, config=cfg, regime="large")


def apply_side_equilibrium_to_metrics(
    metrics: dict[str, Any], decision: SideEquilibriumDecision, *, proposed: TradeDirection
) -> bool:
    """Aplica hard-skip ou soft penalty nas metricas."""
    metrics["side_eq_action"] = decision.action
    metrics["side_eq_reason"] = decision.reason
    metrics["side_eq_call"] = f"{decision.call_wins}/{decision.call_n}"
    metrics["side_eq_put"] = f"{decision.put_wins}/{decision.put_n}"
    metrics["side_eq_freq_bias"] = float(decision.freq_bias)
    metrics["side_eq_side_wr"] = decision.side_wr
    metrics["side_eq_z_vs_half"] = float(decision.z_vs_half)
    metrics["side_eq_proposed"] = proposed.name
    if decision.action == ACTION_HARD_SKIP:
        metrics["gate_reason"] = str(decision.reason or "side_imbalance_small_n")
        metrics["quality_guard_reject"] = True
        metrics["side_eq_kelly_mult"] = 1.0
        metrics["side_eq_margin_boost"] = 0.0
        return True
    if decision.action == ACTION_SOFT:
        metrics["side_eq_kelly_mult"] = float(decision.kelly_mult)
        metrics["side_eq_margin_boost"] = float(decision.margin_boost)
        scale = float(metrics.get("kelly_fraction_scale", 1.0))
        metrics["kelly_fraction_scale"] = scale * float(decision.kelly_mult)
        floor = float(metrics.get("quality_min_direction_margin", 0.0))
        metrics["quality_min_direction_margin"] = floor + float(decision.margin_boost)
        return False
    metrics["side_eq_kelly_mult"] = 1.0
    metrics["side_eq_margin_boost"] = 0.0
    return False


def log_side_equilibrium(
    decision: SideEquilibriumDecision,
    *,
    symbol: str,
    proposed: TradeDirection,
    orch: Any | None = None,
) -> None:
    """Emite log deduplicado SIDE_EQ por ciclo e simbolo."""
    if orch is None:
        return
    cycle = int(getattr(orch, "_active_cycle_id", 0) or 0)
    key = ("side_eq", cycle, str(symbol))
    bag = getattr(orch, "_side_eq_log_keys", None)
    if not isinstance(bag, set):
        orch._side_eq_log_keys = set()
        bag = orch._side_eq_log_keys
    if key in bag:
        return
    bag.add(key)
    logger.info(
        "SIDE_EQ | %s %s | call=%d/%d put=%d/%d | bias=%.2f wr=%s | action=%s",
        symbol,
        proposed.name,
        decision.call_wins,
        decision.call_n,
        decision.put_wins,
        decision.put_n,
        float(decision.freq_bias),
        f"{decision.side_wr:.2f}" if decision.side_wr is not None else "na",
        decision.action,
    )


def resolve_direction_with_side_equilibrium(
    orch: Any | None,
    symbol: str | None,
    proposed: TradeDirection,
    metrics: dict[str, Any],
    *,
    recovery_active: bool = False,
) -> TradeDirection | None:
    """Escolhe lado equilibrado: hard-skip no proposto tenta o oposto; None se ambos bloqueados."""
    waive_hard = bool(recovery_active)
    if bool(metrics.get("side_eq_gate_done")):
        if bool(metrics.get("side_eq_blocked")):
            if not str(metrics.get("gate_reason") or "").strip():
                metrics["gate_reason"] = str(metrics.get("side_eq_reason") or "side_imbalance_both_sides")
                metrics["quality_guard_reject"] = True
            return None
        name = str(metrics.get("exec_direction") or metrics.get("resolved_direction") or proposed.name)
        try:
            return TradeDirection[name]
        except KeyError:
            return proposed
    primary = evaluate_proposed_side_equilibrium(orch, symbol, proposed)
    log_side_equilibrium(primary, symbol=str(symbol or "?"), proposed=proposed, orch=orch)
    if primary.action != ACTION_HARD_SKIP:
        apply_side_equilibrium_to_metrics(metrics, primary, proposed=proposed)
        metrics["side_eq_gate_done"] = True
        metrics["side_eq_blocked"] = False
        metrics.pop("quality_guard_reject", None)
        gate = str(metrics.get("gate_reason") or "")
        if gate.startswith("side_imbalance"):
            metrics.pop("gate_reason", None)
        return proposed
    if positive_meta_edge_keeps_proposed(metrics):
        return soft_keep_proposed(
            metrics,
            primary,
            proposed,
            reason="side_eq_edge_keep_proposed",
            apply_metrics=apply_side_equilibrium_to_metrics,
        )
    apply_side_equilibrium_to_metrics(metrics, primary, proposed=proposed)
    opposite = opposite_trade_direction(proposed)
    alternate = evaluate_proposed_side_equilibrium(orch, symbol, opposite)
    log_side_equilibrium(alternate, symbol=str(symbol or "?"), proposed=opposite, orch=orch)
    if alternate.action == ACTION_HARD_SKIP:
        if waive_hard:
            return soft_keep_proposed(
                metrics,
                primary,
                proposed,
                reason="side_eq_recovery_both_hard",
                apply_metrics=apply_side_equilibrium_to_metrics,
            )
        apply_side_equilibrium_to_metrics(metrics, alternate, proposed=opposite)
        metrics["side_eq_gate_done"] = True
        metrics["side_eq_blocked"] = True
        metrics["gate_reason"] = str(alternate.reason or primary.reason or "side_imbalance_both_sides")
        metrics["quality_guard_reject"] = True
        return None
    toxic_primary = primary_side_is_toxic(primary)
    prefer_alt = alternate_side_is_preferable(primary, alternate, opposite=opposite)
    thin_blocks = thin_margin_blocks_flip(metrics) and not toxic_primary
    if not prefer_alt or thin_blocks:
        if waive_hard:
            keep_reason = "side_eq_recovery_keep" if not prefer_alt else "side_eq_recovery_thin_margin"
            return soft_keep_proposed(
                metrics,
                primary,
                proposed,
                reason=keep_reason,
                apply_metrics=apply_side_equilibrium_to_metrics,
            )
        apply_side_equilibrium_to_metrics(metrics, primary, proposed=proposed)
        metrics["side_eq_gate_done"] = True
        metrics["side_eq_blocked"] = True
        metrics["gate_reason"] = (
            "side_imbalance_thin_margin_flip" if thin_blocks and prefer_alt else "side_imbalance_flip_not_better"
        )
        metrics["quality_guard_reject"] = True
        metrics["side_eq_flip_rejected"] = True
        return None
    if flip_conflicts_price_zone(opposite, metrics) and not toxic_primary:
        if waive_hard:
            return soft_keep_proposed(
                metrics,
                primary,
                proposed,
                reason="side_eq_recovery_zone_keep",
                apply_metrics=apply_side_equilibrium_to_metrics,
            )
        apply_side_equilibrium_to_metrics(metrics, primary, proposed=proposed)
        metrics["side_eq_gate_done"] = True
        metrics["side_eq_blocked"] = True
        metrics["gate_reason"] = "side_imbalance_flip_zone_conflict"
        metrics["quality_guard_reject"] = True
        metrics["side_eq_flip_rejected"] = True
        metrics["side_eq_flip_zone_conflict"] = True
        return None
    if toxic_primary and flip_conflicts_price_zone(opposite, metrics):
        metrics["side_eq_toxic_zone_escape"] = True
    metrics.pop("quality_guard_reject", None)
    gate = str(metrics.get("gate_reason") or "")
    if gate.startswith("side_imbalance"):
        metrics.pop("gate_reason", None)
    metrics["side_eq_flipped"] = True
    metrics["side_eq_flip_from"] = proposed.name
    if toxic_primary:
        metrics["side_eq_toxic_escape"] = True
    apply_side_equilibrium_to_metrics(metrics, alternate, proposed=opposite)
    log_side_eq_flip(orch, symbol=str(symbol or "?"), proposed=proposed, opposite=opposite, reason=primary.reason)
    metrics["exec_direction"] = opposite.name
    metrics["resolved_direction"] = opposite.name
    metrics["side_eq_gate_done"] = True
    metrics["side_eq_blocked"] = False
    return opposite
