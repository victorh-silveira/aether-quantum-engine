"""Modificador de Kelly por divergencia entre ordem e votos tecnicos."""

from __future__ import annotations

from typing import Any

from src.domain.risk.stake_sizing import (
    consensus_entropy_applies_min_stake,
    consensus_entropy_kelly_retention,
)
from src.domain.risk.super_concordance_kelly import is_unanimous_vote_alignment


_REGIME_TACTICAL_INVERT = frozenset({"CLIMAX_EXHAUSTION", "COMPRESSION_TRAP"})


def _regime_tactical_inversion_active(metrics: dict) -> bool:
    """Indica inversao tatica forçada por CLIMAX_EXHAUSTION ou COMPRESSION_TRAP."""
    regime = str(metrics.get("universal_regime") or metrics.get("universal_regime_scenario") or "")
    if regime not in _REGIME_TACTICAL_INVERT:
        return False
    return bool(metrics.get("direction_inverted"))


def _recovery_waives_consensus_penalty(
    metrics: dict,
    kelly_config: dict[str, Any],
    *,
    consecutive_losses: int,
    pending_loss_total: float,
    order_direction: str | None,
) -> bool:
    """Suspende penalidade em recovery com inversao tatica, votos unanimes ou trade_score alto."""
    recovering = float(pending_loss_total) > 0.0 or int(consecutive_losses) > 0
    if not recovering:
        return False
    if _regime_tactical_inversion_active(metrics):
        return True
    if is_unanimous_vote_alignment(
        int(metrics.get("call_votes", 0)),
        int(metrics.get("put_votes", 0)),
        order_direction,
    ):
        return True
    cfg = kelly_config or {}
    score_min = float(cfg.get("penalty_smoothing_trade_score_min", 0.68))
    trade_score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    return trade_score + 1e-9 >= score_min


def consensus_kelly_retention(
    metrics: dict,
    order_direction: str | None,
    *,
    kelly_config: dict[str, Any] | None = None,
    consecutive_losses: int = 0,
    pending_loss_total: float = 0.0,
) -> float:
    """Retorna fator [floor, 1.0] para atenuar f* quando ord diverge do consenso tecnico."""
    if isinstance(metrics, dict):
        recovering = float(pending_loss_total) > 0.0 or int(consecutive_losses) > 0
        if recovering and _recovery_waives_consensus_penalty(
            metrics,
            kelly_config or {},
            consecutive_losses=int(consecutive_losses),
            pending_loss_total=float(pending_loss_total),
            order_direction=order_direction,
        ):
            if _regime_tactical_inversion_active(metrics):
                metrics["consensus_penalty_regime_inversion_waived"] = True
            else:
                metrics["consensus_penalty_recovery_waived"] = True
            return 1.0
    return consensus_entropy_kelly_retention(metrics, order_direction, kelly_config=kelly_config)


__all__ = [
    "consensus_entropy_applies_min_stake",
    "consensus_entropy_kelly_retention",
    "consensus_kelly_retention",
]
