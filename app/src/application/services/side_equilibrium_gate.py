"""Gate de equilibrio CALL/PUT: small-N hard skip, large-N soft Kelly/margem."""

from __future__ import annotations

import logging
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


logger = logging.getLogger("AETH")


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


def log_side_equilibrium(decision: SideEquilibriumDecision, *, symbol: str, proposed: TradeDirection) -> None:
    """Registra telemetria SIDE_EQ no logger AETH."""
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
