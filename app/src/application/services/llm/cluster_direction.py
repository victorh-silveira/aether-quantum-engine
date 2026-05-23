"""Direcao de execucao de clusters US/EU a partir das tags da LLM."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


def cluster_direction_from_tag(cluster_tag: str | None) -> tuple[TradeDirection | None, bool]:
    """Retorna direcao CALL/PUT do cluster sem inversao ou heuristica de conviccao."""
    if cluster_tag not in ("CALL", "PUT"):
        return None, False
    return TradeDirection[cluster_tag], False
