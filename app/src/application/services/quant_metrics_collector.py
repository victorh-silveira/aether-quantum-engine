"""Coletor de telemetria quant e observabilidade operacional em tempo real."""

from __future__ import annotations

import math
from collections import deque
from typing import Any


class QuantMetricsCollector:
    """Acumula e sintetiza metricas quantitativas de execucao, calibracao e risco."""

    def __init__(self, initial_balance: float = 0.0, brier_window_size: int = 20) -> None:
        """Inicializa historicos circulares de latencia, brier score e balanco."""
        self._initial_balance: float = float(initial_balance)
        self._current_balance: float = float(initial_balance)
        self._high_water_mark: float = float(initial_balance)
        self._max_drawdown_pct: float = 0.0
        self._brier_window_size: int = max(5, brier_window_size)

        self._tick_to_order_latencies: deque[float] = deque(maxlen=100)
        self._ping_rtts: deque[float] = deque(maxlen=100)
        self._trade_probabilities: deque[float] = deque(maxlen=self._brier_window_size)
        self._trade_outcomes: deque[int] = deque(maxlen=self._brier_window_size)

        self._total_cycles: int = 0
        self._total_executions: int = 0
        self._total_skips: int = 0
        self._total_wins: int = 0
        self._total_losses: int = 0
        self._total_pnl: float = 0.0

    def record_cycle(self, *, is_execution: bool, reason: str = "ok") -> None:
        """Registra a conclusao de um ciclo de analise do orquestrador."""
        _ = reason
        self._total_cycles += 1
        if is_execution:
            self._total_executions += 1
        else:
            self._total_skips += 1

    def has_observations(self) -> bool:
        """Indica se o coletor recebeu dados reais desta sessao."""
        return bool(
            self._initial_balance > 0.0
            or self._total_cycles
            or self._total_wins
            or self._total_losses
            or self._tick_to_order_latencies
            or self._ping_rtts
        )

    def record_latency(self, tick_to_order_ms: float, ping_rtt_ms: float | None = None) -> None:
        """Armazena medidas pontuais de latencia de processamento e handshake."""
        if math.isfinite(tick_to_order_ms) and tick_to_order_ms >= 0.0:
            self._tick_to_order_latencies.append(float(tick_to_order_ms))
        if ping_rtt_ms is not None and math.isfinite(ping_rtt_ms) and ping_rtt_ms >= 0.0:
            self._ping_rtts.append(float(ping_rtt_ms))

    def record_trade_outcome(self, *, predicted_prob: float, is_win: bool, profit_usd: float) -> None:
        """Processa a liquidacao do contrato, atualizando calibracao e drawdown."""
        prob = max(0.0, min(1.0, float(predicted_prob)))
        outcome = 1 if is_win else 0
        self._trade_probabilities.append(prob)
        self._trade_outcomes.append(outcome)

        pnl = float(profit_usd)
        self._total_pnl += pnl
        self._current_balance += pnl

        if is_win:
            self._total_wins += 1
        else:
            self._total_losses += 1

        self._high_water_mark = max(self._high_water_mark, self._current_balance)

        if self._high_water_mark > 0.0:
            dd_pct = (self._high_water_mark - self._current_balance) / self._high_water_mark * 100.0
            self._max_drawdown_pct = max(self._max_drawdown_pct, dd_pct)

    def compute_brier_score(self) -> float | None:
        """Calcula o Brier Score medio sobre a janela deslizante de liquidacoes."""
        if not self._trade_outcomes:
            return None
        total_sq_error = sum(
            (p - y) ** 2 for p, y in zip(self._trade_probabilities, self._trade_outcomes, strict=False)
        )
        return total_sq_error / len(self._trade_outcomes)

    def get_summary(self) -> dict[str, Any]:
        """Gera sumario consolidado para relatorios e telemetria operacional."""
        avg_tick_order = (
            sum(self._tick_to_order_latencies) / len(self._tick_to_order_latencies)
            if self._tick_to_order_latencies
            else 0.0
        )
        avg_ping = sum(self._ping_rtts) / len(self._ping_rtts) if self._ping_rtts else 0.0
        acceptance_rate = (self._total_executions / self._total_cycles * 100.0) if self._total_cycles > 0 else 0.0
        win_rate = (
            (self._total_wins / (self._total_wins + self._total_losses) * 100.0)
            if (self._total_wins + self._total_losses) > 0
            else 0.0
        )

        return {
            "total_cycles": self._total_cycles,
            "total_executions": self._total_executions,
            "total_skips": self._total_skips,
            "gate_acceptance_rate_pct": round(acceptance_rate, 2),
            "total_wins": self._total_wins,
            "total_losses": self._total_losses,
            "win_rate_pct": round(win_rate, 2),
            "total_pnl_usd": round(self._total_pnl, 4),
            "current_balance": round(self._current_balance, 2),
            "max_drawdown_pct": round(self._max_drawdown_pct, 2),
            "rolling_brier_score": round(self.compute_brier_score() or 0.0, 4),
            "avg_tick_to_order_ms": round(avg_tick_order, 2),
            "avg_ping_rtt_ms": round(avg_ping, 2),
        }

    def reset(self, initial_balance: float | None = None) -> None:
        """Reinicia os acumuladores mantendo o tamanho das janelas configuradas."""
        if initial_balance is not None:
            self._initial_balance = float(initial_balance)
        self._current_balance = self._initial_balance
        self._high_water_mark = self._initial_balance
        self._max_drawdown_pct = 0.0
        self._tick_to_order_latencies.clear()
        self._ping_rtts.clear()
        self._trade_probabilities.clear()
        self._trade_outcomes.clear()
        self._total_cycles = 0
        self._total_executions = 0
        self._total_skips = 0
        self._total_wins = 0
        self._total_losses = 0
        self._total_pnl = 0.0
