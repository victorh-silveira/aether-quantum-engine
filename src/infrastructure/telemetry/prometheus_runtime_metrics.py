"""Estado e renderizacao de metricas operacionais estendidas Prometheus."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeTelemetryState:
    """Acumula latencia de ciclos, bloqueios, RTT Deriv, Stop-Win e payout real."""

    last_cycle_duration_seconds: float | None = None
    last_history_sync_seconds: float | None = None
    stop_win_reached: bool = False
    cycle_skips: dict[str, int] = field(default_factory=dict)
    deriv_rtt_ms: dict[str, float] = field(default_factory=dict)
    deriv_errors: dict[str, int] = field(default_factory=dict)
    quote_payout_rate: dict[str, float] = field(default_factory=dict)
    quote_breakeven_prob: dict[str, float] = field(default_factory=dict)

    def record_cycle_duration(self, duration_seconds: float) -> None:
        """Atualiza duracao do ultimo ciclo de avaliacao/execucao."""
        self.last_cycle_duration_seconds = max(0.0, float(duration_seconds))

    def record_cycle_skip(self, reason: str) -> None:
        """Incrementa contador de ciclo ignorado por guarda operacional."""
        key = str(reason or "unknown")
        self.cycle_skips[key] = int(self.cycle_skips.get(key, 0)) + 1

    def record_deriv_rtt(self, method: str, rtt_ms: float, *, ok: bool = True) -> None:
        """Registra latencia da API Deriv e eventuais falhas por metodo."""
        key = str(method or "unknown")
        self.deriv_rtt_ms[key] = max(0.0, float(rtt_ms))
        if not ok:
            self.deriv_errors[key] = int(self.deriv_errors.get(key, 0)) + 1

    def record_quote_payout(self, symbol: str, price: float, payout: float) -> None:
        """Registra retorno liquido percentual e break-even implicito observados na corretora."""
        if price <= 0.0 or payout <= price:
            return
        sym = str(symbol)
        self.quote_payout_rate[sym] = (float(payout) - float(price)) / float(price)
        self.quote_breakeven_prob[sym] = float(price) / float(payout)

    def render_prometheus_lines(self) -> list[str]:
        """Formata linhas Prometheus com nome, tipo e labels padrao Aether."""
        lines = [
            "# TYPE aether_session_stop_win_reached gauge",
            f"aether_session_stop_win_reached {int(self.stop_win_reached)}",
        ]
        if self.last_cycle_duration_seconds is not None:
            lines.append("# TYPE aether_cycle_duration_seconds gauge")
            lines.append(f"aether_cycle_duration_seconds {self.last_cycle_duration_seconds:.6f}")
        if self.last_history_sync_seconds is not None:
            lines.append("# TYPE aether_history_sync_duration_seconds gauge")
            lines.append(f"aether_history_sync_duration_seconds {self.last_history_sync_seconds:.3f}")
        if self.cycle_skips:
            lines.append("# TYPE aether_cycle_skips_total counter")
            for reason, total in sorted(self.cycle_skips.items()):
                lines.append(f'aether_cycle_skips_total{{reason="{reason}"}} {total}')
        if self.deriv_rtt_ms:
            lines.append("# TYPE aether_deriv_ws_rtt_ms gauge")
            for method, rtt_ms in sorted(self.deriv_rtt_ms.items()):
                lines.append(f'aether_deriv_ws_rtt_ms{{method="{method}"}} {rtt_ms:.3f}')
        if self.deriv_errors:
            lines.append("# TYPE aether_deriv_ws_errors_total counter")
            for method, total in sorted(self.deriv_errors.items()):
                lines.append(f'aether_deriv_ws_errors_total{{method="{method}"}} {total}')
        if self.quote_payout_rate:
            lines.append("# TYPE aether_quote_observed_payout_rate gauge")
            lines.append("# TYPE aether_quote_observed_breakeven_prob gauge")
            for sym in sorted(self.quote_payout_rate):
                lines.append(f'aether_quote_observed_payout_rate{{symbol="{sym}"}} {self.quote_payout_rate[sym]:.6f}')
                lines.append(
                    f'aether_quote_observed_breakeven_prob{{symbol="{sym}"}} {self.quote_breakeven_prob[sym]:.6f}'
                )
        return lines
