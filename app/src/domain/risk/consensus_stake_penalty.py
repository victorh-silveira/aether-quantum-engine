"""Modificador de Kelly por divergencia entre ordem e votos tecnicos."""

from __future__ import annotations

from typing import Any

from src.domain.risk.stake_sizing import (
    consensus_entropy_applies_min_stake,
    consensus_entropy_kelly_retention,
)


def _smooth_recovery_penalty(
    retention: float,
    metrics: dict,
    kelly_config: dict[str, Any],
    *,
    consecutive_losses: int,
    pending_loss_total: float,
) -> float:
    """Amortece penalidade convexa em recovery quando trade_score e estavel."""
    if retention >= 1.0:
        return retention
    recovering = float(pending_loss_total) > 0.0 or int(consecutive_losses) > 0
    if not recovering:
        return retention
    cfg = kelly_config or {}
    score_min = float(cfg.get("penalty_smoothing_trade_score_min", 0.70))
    trade_score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    if trade_score + 1e-9 <= score_min:
        return retention
    factor = float(cfg.get("penalty_smoothing_factor", 0.40))
    cut = 1.0 - float(retention)
    return min(1.0, float(retention) + cut * factor)


def consensus_kelly_retention(
    metrics: dict,
    order_direction: str | None,
    *,
    kelly_config: dict[str, Any] | None = None,
    consecutive_losses: int = 0,
    pending_loss_total: float = 0.0,
) -> float:
    """Retorna fator [floor, 1.0] para atenuar f* quando ord diverge do consenso tecnico."""
    raw = consensus_entropy_kelly_retention(metrics, order_direction, kelly_config=kelly_config)
    if not isinstance(metrics, dict):
        return raw
    smoothed = _smooth_recovery_penalty(
        raw,
        metrics,
        kelly_config or {},
        consecutive_losses=int(consecutive_losses),
        pending_loss_total=float(pending_loss_total),
    )
    if smoothed > raw + 1e-9:
        metrics["consensus_penalty_smoothed"] = True
        metrics["consensus_entropy_retention_raw"] = raw
    return smoothed


__all__ = [
    "consensus_entropy_applies_min_stake",
    "consensus_entropy_kelly_retention",
    "consensus_kelly_retention",
]
