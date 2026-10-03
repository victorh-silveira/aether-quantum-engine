"""Testes unitarios para o formatador de metricas Prometheus."""

from src.application.services.quant_metrics_collector import QuantMetricsCollector
from src.infrastructure.telemetry.prometheus_exporter import PrometheusMetricsExporter


def test_prometheus_metrics_exporter_format():
    collector = QuantMetricsCollector(initial_balance=1000.0)
    collector.record_cycle(is_execution=True)
    collector.record_latency(tick_to_order_ms=12.5, ping_rtt_ms=45.0)
    collector.record_trade_outcome(predicted_prob=0.65, is_win=True, profit_usd=85.0)

    payload = PrometheusMetricsExporter.format_metrics(collector)
    assert isinstance(payload, str)
    assert "aether_current_balance_usd 1085.0" in payload
    assert "aether_total_executions_total 1" in payload
    assert "aether_total_wins_total 1" in payload
    assert "aether_avg_tick_to_order_ms 12.5" in payload
    assert "aether_avg_ping_rtt_ms 45.0" in payload
    assert payload.endswith("\n")
