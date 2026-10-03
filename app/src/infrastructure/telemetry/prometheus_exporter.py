"""Formatador e exportador de metricas no padrao Prometheus e OpenMetrics."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.application.services.quant_metrics_collector import QuantMetricsCollector


class PrometheusMetricsExporter:
    """Serializa telemetria quantitativa do coletor em formato OpenMetrics texto."""

    @staticmethod
    def format_metrics(collector: QuantMetricsCollector) -> str:
        """Gera payload textual compativel com Prometheus scrape."""
        summary = collector.get_summary()

        lines: list[str] = [
            "aether_current_balance_usd " + str(summary.get("current_balance", 0.0)),
            "aether_max_drawdown_pct " + str(summary.get("max_drawdown_pct", 0.0)),
            "aether_win_rate_pct " + str(summary.get("win_rate_pct", 0.0)),
            "aether_rolling_brier_score " + str(summary.get("rolling_brier_score", 0.0)),
            "aether_avg_tick_to_order_ms " + str(summary.get("avg_tick_to_order_ms", 0.0)),
            "aether_avg_ping_rtt_ms " + str(summary.get("avg_ping_rtt_ms", 0.0)),
            "aether_gate_acceptance_rate_pct " + str(summary.get("gate_acceptance_rate_pct", 0.0)),
            "aether_total_cycles_total " + str(summary.get("total_cycles", 0)),
            "aether_total_executions_total " + str(summary.get("total_executions", 0)),
            "aether_total_skips_total " + str(summary.get("total_skips", 0)),
            "aether_total_wins_total " + str(summary.get("total_wins", 0)),
            "aether_total_losses_total " + str(summary.get("total_losses", 0)),
            "aether_total_pnl_usd " + str(summary.get("total_pnl_usd", 0.0)),
        ]
        return "\n".join(lines) + "\n"
