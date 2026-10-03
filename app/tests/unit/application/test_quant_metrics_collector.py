"""Testes unitarios para o coletor de metricas quantitativas e observabilidade."""

import pytest

from src.application.services.quant_metrics_collector import QuantMetricsCollector


def test_quant_metrics_collector_initial_state():
    collector = QuantMetricsCollector(initial_balance=1000.0)
    summary = collector.get_summary()

    assert summary["total_cycles"] == 0
    assert summary["total_executions"] == 0
    assert summary["total_skips"] == 0
    assert summary["current_balance"] == 1000.0
    assert summary["max_drawdown_pct"] == 0.0
    assert summary["rolling_brier_score"] == 0.0
    assert collector.compute_brier_score() is None


def test_quant_metrics_collector_cycle_recording():
    collector = QuantMetricsCollector()
    collector.record_cycle(is_execution=True)
    collector.record_cycle(is_execution=False)
    collector.record_cycle(is_execution=False)

    summary = collector.get_summary()
    assert summary["total_cycles"] == 3
    assert summary["total_executions"] == 1
    assert summary["total_skips"] == 2
    assert summary["gate_acceptance_rate_pct"] == pytest.approx(33.33, abs=0.01)


def test_quant_metrics_collector_latency_recording():
    collector = QuantMetricsCollector()
    collector.record_latency(tick_to_order_ms=12.5, ping_rtt_ms=45.0)
    collector.record_latency(tick_to_order_ms=17.5, ping_rtt_ms=55.0)
    collector.record_latency(tick_to_order_ms=-5.0, ping_rtt_ms=float("nan"))

    summary = collector.get_summary()
    assert summary["avg_tick_to_order_ms"] == 15.0
    assert summary["avg_ping_rtt_ms"] == 50.0


def test_quant_metrics_collector_trade_outcomes_and_brier():
    collector = QuantMetricsCollector(initial_balance=1000.0, brier_window_size=5)

    collector.record_trade_outcome(predicted_prob=0.60, is_win=True, profit_usd=85.0)
    collector.record_trade_outcome(predicted_prob=0.70, is_win=False, profit_usd=-100.0)

    expected_brier = ((0.60 - 1.0) ** 2 + (0.70 - 0.0) ** 2) / 2.0
    assert collector.compute_brier_score() == pytest.approx(expected_brier, abs=1e-5)

    summary = collector.get_summary()
    assert summary["total_wins"] == 1
    assert summary["total_losses"] == 1
    assert summary["win_rate_pct"] == 50.0
    assert summary["total_pnl_usd"] == -15.0
    assert summary["current_balance"] == 985.0

    peak = 1085.0
    trough = 985.0
    expected_dd = (peak - trough) / peak * 100.0
    assert summary["max_drawdown_pct"] == pytest.approx(expected_dd, abs=0.01)


def test_quant_metrics_collector_reset():
    collector = QuantMetricsCollector(initial_balance=500.0)
    collector.record_cycle(is_execution=True)
    collector.record_trade_outcome(predicted_prob=0.8, is_win=True, profit_usd=40.0)

    collector.reset(initial_balance=600.0)
    summary = collector.get_summary()

    assert summary["total_cycles"] == 0
    assert summary["total_wins"] == 0
    assert summary["current_balance"] == 600.0
    assert summary["max_drawdown_pct"] == 0.0
