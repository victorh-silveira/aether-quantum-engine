"""Modelo de dominio para decisao binaria continua produzida pelo Laya."""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.models.trade import TradeDirection


@dataclass(frozen=True, slots=True)
class LayaDecision:
    """Decisao binaria calibrada (CALL ou PUT) sem SKIP produzida pelo Laya."""

    symbol: str
    direction: TradeDirection
    probability: float
    prob_call: float
    prob_put: float
    confidence: float
    latency_ms: float
    indicators: dict[str, float]

    def __post_init__(self) -> None:
        """Valida que a direcao e estritamente binaria (CALL ou PUT)."""
        if self.direction not in (TradeDirection.CALL, TradeDirection.PUT):
            raise ValueError(f"LayaDecision exige CALL ou PUT continuo, recebido {self.direction!r}")
