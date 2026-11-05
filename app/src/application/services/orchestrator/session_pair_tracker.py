"""Rastreador de performance e saude de pares ao longo da sessao do bot."""

from __future__ import annotations

import time
from typing import Any


class SessionPairTracker:
    """Gerencia estatisticas e pesos elasticos de rotacao por ativo na sessao ativa."""

    def __init__(self) -> None:
        """Inicializa as estruturas em memoria do rastreador."""
        self._stats: dict[str, dict[str, Any]] = {}

    def record_symbol_settlement(self, symbol: str, *, won: bool, profit: float = 0.0) -> None:
        """Registra o desfecho de liquidacao de uma ordem para o ativo."""
        sym = str(symbol).upper().strip()
        data = self._stats.setdefault(
            sym,
            {"wins": 0, "losses": 0, "streak": 0, "pnl": 0.0, "last_settled": 0.0},
        )
        data["pnl"] = float(data.get("pnl", 0.0)) + float(profit)
        data["last_settled"] = time.time()

        if won:
            data["wins"] = int(data.get("wins", 0)) + 1
            cur_streak = int(data.get("streak", 0))
            data["streak"] = (cur_streak + 1) if cur_streak > 0 else 1
        else:
            data["losses"] = int(data.get("losses", 0)) + 1
            cur_streak = int(data.get("streak", 0))
            data["streak"] = (cur_streak - 1) if cur_streak < 0 else -1

    def symbol_rotation_adjustment(self, symbol: str) -> float:
        """Calcula o ajuste elastico no score de rotacao do par."""
        sym = str(symbol).upper().strip()
        data = self._stats.get(sym)
        if not data:
            return 0.0
        streak = int(data.get("streak", 0))
        if streak < 0:
            return max(-0.25, float(streak) * 0.10)
        return min(0.12, float(streak) * 0.04) if streak > 0 else 0.0

    def get_symbol_stats(self, symbol: str) -> dict[str, Any]:
        """Retorna uma copia das estatisticas do par na sessao."""
        sym = str(symbol).upper().strip()
        return dict(
            self._stats.get(
                sym,
                {"wins": 0, "losses": 0, "streak": 0, "pnl": 0.0, "last_settled": 0.0},
            )
        )

    def reset(self) -> None:
        """Reseta o estado da sessao."""
        self._stats.clear()
