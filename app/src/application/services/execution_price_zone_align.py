"""Alinhamento simples de direcao a price zone."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


def align_direction_to_price_zone(direction: TradeDirection, metrics: dict[str, Any]) -> TradeDirection:
    """Substitui o lado pelo da zona quando price_zone_direction estiver setado."""
    if bool(metrics.get("side_eq_flipped")):
        return direction
    raw = str(metrics.get("price_zone_direction") or "").upper()
    if raw == TradeDirection.CALL.name:
        return TradeDirection.CALL
    if raw == TradeDirection.PUT.name:
        return TradeDirection.PUT
    return direction
