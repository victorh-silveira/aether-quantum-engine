"""Teto soberano para checkpoints TCN sem qualificacao estatistica."""

from typing import Any


MAX_CHECKPOINT_STAKE_PCT = 0.01


def checkpoint_stake_cap_pct(metrics: dict[str, Any]) -> float:
    """Limita a fracao solicitada ao maximo operacional de 1%."""
    return min(MAX_CHECKPOINT_STAKE_PCT, max(0.0, float(metrics.get("provisional_max_stake_pct", 0.0) or 0.0)))
