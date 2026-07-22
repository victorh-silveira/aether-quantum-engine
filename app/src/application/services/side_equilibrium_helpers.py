"""Helpers do gate SIDE_EQ: keep, flip e toxicidade."""

from __future__ import annotations

import logging
from typing import Any

from src.domain.analytics.side_equilibrium import (
    ACTION_HARD_SKIP,
    ACTION_SOFT,
    SideEquilibriumDecision,
)
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")


def positive_meta_edge_keeps_proposed(metrics: dict[str, Any]) -> bool:
    """True quando o edge meta positivo deve preservar o lado TCN/meta sem flip."""
    edge = metrics.get("predicted_payoff_edge")
    if edge is None:
        return False
    return float(edge) > 0.0


def thin_margin_blocks_flip(metrics: dict[str, Any]) -> bool:
    """True quando a margem TCN e fina demais para justificar flip SIDE_EQ."""
    raw = metrics.get("direction_margin")
    if raw is None:
        return False
    try:
        return float(raw) + 1e-12 < 0.02
    except (TypeError, ValueError):
        return False


def soft_keep_proposed(
    metrics: dict[str, Any],
    primary: SideEquilibriumDecision,
    proposed: TradeDirection,
    *,
    reason: str,
    apply_metrics,
) -> TradeDirection:
    """Mantem o lado proposto com soft penalty em vez de bloquear ou flipar."""
    kelly = float(primary.kelly_mult) if float(primary.kelly_mult) > 0.0 else 0.55
    kelly = max(0.35, min(0.85, kelly if kelly < 1.0 else 0.55))
    soft = SideEquilibriumDecision(
        action=ACTION_SOFT,
        reason=str(reason),
        kelly_mult=kelly,
        margin_boost=max(0.0, float(primary.margin_boost)),
        call_n=primary.call_n,
        call_wins=primary.call_wins,
        put_n=primary.put_n,
        put_wins=primary.put_wins,
        side_wr=primary.side_wr,
        freq_bias=primary.freq_bias,
        z_vs_half=primary.z_vs_half,
    )
    apply_metrics(metrics, soft, proposed=proposed)
    metrics.pop("quality_guard_reject", None)
    gate = str(metrics.get("gate_reason") or "")
    if gate.startswith("side_imbalance"):
        metrics.pop("gate_reason", None)
    metrics[str(reason)] = True
    metrics["side_eq_gate_done"] = True
    metrics["side_eq_blocked"] = False
    metrics["exec_direction"] = proposed.name
    metrics["resolved_direction"] = proposed.name
    return proposed


def primary_side_is_toxic(primary: SideEquilibriumDecision) -> bool:
    """True quando o lado primario esta em hard-skip toxico por WR baixo."""
    if primary.action != ACTION_HARD_SKIP:
        return False
    if primary.side_wr is None:
        return True
    return float(primary.side_wr) + 1e-12 < 0.40


def flip_conflicts_price_zone(opposite: TradeDirection, metrics: dict[str, Any]) -> bool:
    """True quando o flip proposto conflita com a price zone ativa."""
    zone_side = str(metrics.get("price_zone_direction") or "").upper()
    if zone_side not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        return False
    return opposite.name != zone_side


def alternate_side_is_preferable(
    primary: SideEquilibriumDecision,
    alternate: SideEquilibriumDecision,
    *,
    opposite: TradeDirection,
) -> bool:
    """Exige amostras e WR claros no alternativo antes de flipar o lado TCN."""
    alt_wr = alternate.side_wr
    pri_wr = primary.side_wr
    alt_n = int(alternate.call_n if opposite == TradeDirection.CALL else alternate.put_n)
    if alt_n < 2 or alt_wr is None:
        return False
    if float(alt_wr) + 1e-12 < 0.50:
        return False
    if pri_wr is None:
        return True
    return float(alt_wr) + 1e-12 >= float(pri_wr) + 0.10


def log_side_eq_flip(
    orch: Any | None,
    *,
    symbol: str,
    proposed: TradeDirection,
    opposite: TradeDirection,
    reason: str | None,
) -> None:
    """Registra flip SIDE_EQ deduplicado por ciclo."""
    cycle = int(getattr(orch, "_active_cycle_id", 0) or 0) if orch is not None else 0
    key = ("flip", cycle, symbol, proposed.name, opposite.name, str(reason or "side_imbalance"))
    if orch is not None:
        bag = getattr(orch, "_side_eq_log_keys", None)
        if not isinstance(bag, set):
            orch._side_eq_log_keys = set()
            bag = orch._side_eq_log_keys
        if key in bag:
            return
        bag.add(key)
    logger.info(
        "SIDE_EQ_FLIP | %s %s -> %s | because=%s",
        symbol,
        proposed.name,
        opposite.name,
        str(reason or "side_imbalance"),
    )
