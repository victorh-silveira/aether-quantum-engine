"""Instrumentacao automatica de metricas de negocio seguindo o padrao OpenTelemetry."""

from __future__ import annotations

import importlib
import logging


logger = logging.getLogger("AETH")

try:
    otel_metrics = importlib.import_module("opentelemetry.metrics")
    _OTEL_SDK_AVAILABLE = True
except ImportError:
    otel_metrics = None
    _OTEL_SDK_AVAILABLE = False


class BusinessMetricsInstrumentor:
    """Instrumentador semantico de metricas de negocio para trading quantitativo."""

    def __init__(self, service_name: str = "aether_quantum_engine") -> None:
        """Inicializa medidores compativeis com OpenTelemetry e buffers locais."""
        self._service_name = service_name
        self._current_balance: float = 0.0
        self._total_pnl: float = 0.0
        self._high_water_mark: float = 0.0
        self._max_drawdown_pct: float = 0.0
        self._recent_probs: list[float] = []
        self._recent_outcomes: list[int] = []
        self._contract_counts: dict[tuple[str, str, str], int] = {}
        self._gate_verdicts: dict[tuple[str, str, str], int] = {}

        self._meter = None
        if _OTEL_SDK_AVAILABLE and otel_metrics is not None:
            try:
                self._meter = otel_metrics.get_meter(self._service_name)
            except Exception as exc:
                logger.debug("OTEL: Falha ao obter meter: %s", exc)
                self._meter = None

    @staticmethod
    def is_otel_sdk_available() -> bool:
        """Indica se a biblioteca oficial OpenTelemetry Metrics esta instalada."""
        return _OTEL_SDK_AVAILABLE

    def update_balance(self, balance_usd: float) -> None:
        """Atualiza a banca operacional e recalcula o drawdown instantaneo."""
        val = float(balance_usd)
        self._current_balance = val
        self._high_water_mark = max(self._high_water_mark, val)
        if self._high_water_mark > 0.0:
            dd = (self._high_water_mark - val) / self._high_water_mark * 100.0
            self._max_drawdown_pct = max(self._max_drawdown_pct, dd)

    def record_cycle(
        self,
        symbol: str,
        *,
        is_execution: bool,
        reason: str = "ok",
        duration_ms: float = 0.0,
    ) -> None:
        """Registra automaticamente conclusao do ciclo com veredito de gates."""
        _ = duration_ms
        verdict = "EXEC" if is_execution else "SKIP"
        clean_reason = str(reason).strip() or "ok"
        key = (str(symbol), verdict, clean_reason)
        self._gate_verdicts[key] = self._gate_verdicts.get(key, 0) + 1

    def record_trade(
        self,
        symbol: str,
        direction: str,
        *,
        is_win: bool,
        profit_usd: float,
        predicted_prob: float,
    ) -> None:
        """Registra liquidacao contratual atualizando PnL e Brier Score de negocio."""
        outcome = "WIN" if is_win else "LOSS"
        dir_name = str(direction).upper()
        sym_name = str(symbol)
        key = (sym_name, dir_name, outcome)
        self._contract_counts[key] = self._contract_counts.get(key, 0) + 1

        pnl = float(profit_usd)
        self._total_pnl += pnl
        self.update_balance(self._current_balance + pnl)

        self._recent_probs.append(max(0.0, min(1.0, float(predicted_prob))))
        self._recent_outcomes.append(1 if is_win else 0)
        if len(self._recent_outcomes) > 20:
            self._recent_probs.pop(0)
            self._recent_outcomes.pop(0)

    def compute_rolling_brier_score(self) -> float:
        """Calcula o Brier Score de calibração sobre os últimos 20 contratos."""
        if not self._recent_outcomes:
            return 0.0
        sq_errors = [(p - y) ** 2 for p, y in zip(self._recent_probs, self._recent_outcomes, strict=False)]
        return float(sum(sq_errors) / len(sq_errors))

    def format_prometheus_metrics(self) -> str:
        """Serializa todas as metricas de negocio no formato OpenMetrics com labels semanticos."""
        lines: list[str] = [
            f"aether_trading_balance_usd {round(self._current_balance, 2)}",
            f"aether_trading_pnl_usd {round(self._total_pnl, 4)}",
            f"aether_trading_max_drawdown_pct {round(self._max_drawdown_pct, 2)}",
            f"aether_trading_brier_score {round(self.compute_rolling_brier_score(), 4)}",
        ]

        for (sym, direction, outcome), count in self._contract_counts.items():
            line = (
                f'aether_trading_contracts_total{{symbol="{sym}",direction="{direction}",outcome="{outcome}"}} {count}'
            )
            lines.append(line)

        for (sym, verdict, reason), count in self._gate_verdicts.items():
            line = f'aether_trading_gate_verdicts_total{{symbol="{sym}",verdict="{verdict}",reason="{reason}"}} {count}'
            lines.append(line)

        return "\n".join(lines) + "\n"
