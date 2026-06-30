"""Modificador de Kelly por divergencia entre ordem e votos tecnicos."""

from __future__ import annotations

from typing import Any

from src.domain.risk.stake_sizing import (
    consensus_entropy_applies_min_stake,
    consensus_entropy_kelly_retention,
)


def consensus_kelly_retention(
    metrics: dict,
    order_direction: str | None,
    *,
    kelly_config: dict[str, Any] | None = None,
) -> float:
    """Retorna fator [floor, 1.0] para atenuar f* quando ord diverge do consenso tecnico."""
    return consensus_entropy_kelly_retention(metrics, order_direction, kelly_config=kelly_config)


__all__ = [
    "consensus_entropy_applies_min_stake",
    "consensus_entropy_kelly_retention",
    "consensus_kelly_retention",
]
