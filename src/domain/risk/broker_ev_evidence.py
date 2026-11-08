"""Qualificacao economica por contratos DEMO confirmados pelo broker."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BrokerEvEvidence:
    """Amostra de uma versao do indicador em um unico ativo."""

    purchased: int
    returns: tuple[float, ...]
    minimum_settled: int = 200
    exploratory_purchased: int = 0

    @property
    def settled(self) -> int:
        """Quantidade liquidada pelo broker."""
        return len(self.returns)

    @property
    def mean_return(self) -> float | None:
        """Retorno liquido medio por unidade de stake."""
        return sum(self.returns) / self.settled if self.settled else None

    @property
    def lower_95(self) -> float | None:
        """Limite inferior normal unilateral da media amostral."""
        if self.settled < 2 or any(not math.isfinite(value) for value in self.returns):
            return None
        mean = self.mean_return
        variance = sum((value - mean) ** 2 for value in self.returns) / (self.settled - 1)
        return mean - 1.653 * math.sqrt(variance / self.settled)

    @property
    def promoted(self) -> bool:
        """Habilita a versao apos amostra e retorno conservador positivos."""
        lower = self.lower_95
        return self.settled >= self.minimum_settled and lower is not None and lower > 0.0

    @property
    def pilot_available(self) -> bool:
        """Mantem o piloto ate completar compras confirmatorias, independentemente da exploracao."""
        return self.purchased < self.minimum_settled

    @property
    def posterior_win_probability(self) -> float | None:
        """Media da posterior Jeffreys para vitorias confirmadas."""
        if self.settled < self.minimum_settled:
            return None
        wins = sum(value > 0 for value in self.returns)
        return (wins + 0.5) / (self.settled + 1.0)

    @property
    def suspended(self) -> bool:
        """Suspende REAL somente apos janela recente completa e limite nao positivo."""
        lower = self.lower_95
        return self.settled >= self.minimum_settled and lower is not None and lower <= 0.0
