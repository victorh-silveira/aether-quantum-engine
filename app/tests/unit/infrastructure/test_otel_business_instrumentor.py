"""Testes unitarios para o instrumentador automatico de metricas de negocio OpenTelemetry."""

import src.infrastructure.telemetry.otel_business_instrumentor as otel_mod
from src.infrastructure.telemetry.otel_business_instrumentor import BusinessMetricsInstrumentor


def test_business_metrics_instrumentor_balance_and_drawdown():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.update_balance(10000.0)
    assert instrumentor._current_balance == 10000.0
    assert instrumentor._max_drawdown_pct == 0.0

    instrumentor.update_balance(9500.0)
    assert instrumentor._max_drawdown_pct == 5.0

    instrumentor.update_balance(10500.0)
    assert instrumentor._high_water_mark == 10500.0
    assert instrumentor._max_drawdown_pct == 5.0


def test_business_metrics_instrumentor_trade_and_brier():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.update_balance(1000.0)

    instrumentor.record_trade("1HZ75V", "CALL", is_win=True, profit_usd=85.0, predicted_prob=0.60)
    assert instrumentor._total_pnl == 85.0
    assert instrumentor._current_balance == 1085.0

    brier = instrumentor.compute_rolling_brier_score()
    expected_brier = (0.60 - 1.0) ** 2
    assert abs(brier - expected_brier) < 1e-4


def test_business_metrics_instrumentor_cycle_recording():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.record_cycle("1HZ75V", is_execution=True)
    instrumentor.record_cycle("1HZ75V", is_execution=False, reason="quote_edge_below_min")

    payload = instrumentor.format_prometheus_metrics()
    assert 'aether_trading_gate_verdicts_total{symbol="1HZ75V",verdict="EXEC",reason="ok"} 1' in payload
    assert (
        'aether_trading_gate_verdicts_total{symbol="1HZ75V",verdict="SKIP",reason="quote_edge_below_min"} 1' in payload
    )


def test_business_metrics_instrumentor_is_otel_sdk_available():
    available = BusinessMetricsInstrumentor.is_otel_sdk_available()
    assert isinstance(available, bool)


def test_business_metrics_instrumentor_sliding_window_pop():
    instrumentor = BusinessMetricsInstrumentor()
    for i in range(25):
        instrumentor.record_trade("1HZ75V", "CALL", is_win=(i % 2 == 0), profit_usd=10.0, predicted_prob=0.55)
    assert len(instrumentor._recent_probs) == 20
    assert len(instrumentor._recent_outcomes) == 20


def test_business_metrics_instrumentor_with_mock_otel(monkeypatch):
    class MockOtelMetrics:
        @staticmethod
        def get_meter(name: str):
            _ = name
            return "mock_meter"

    monkeypatch.setattr(otel_mod, "_OTEL_SDK_AVAILABLE", True)
    monkeypatch.setattr(otel_mod, "otel_metrics", MockOtelMetrics)

    instrumentor = BusinessMetricsInstrumentor()
    assert instrumentor._meter == "mock_meter"


def test_business_metrics_instrumentor_with_mock_otel_exception(monkeypatch):
    class BrokenOtelMetrics:
        @staticmethod
        def get_meter(name: str):
            _ = name
            raise RuntimeError("Erro ao instanciar meter")

    monkeypatch.setattr(otel_mod, "_OTEL_SDK_AVAILABLE", True)
    monkeypatch.setattr(otel_mod, "otel_metrics", BrokenOtelMetrics)

    instrumentor = BusinessMetricsInstrumentor()
    assert instrumentor._meter is None


def test_business_metrics_instrumentor_reload_with_mock(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    mock_otel = ModuleType("opentelemetry")
    mock_metrics = ModuleType("opentelemetry.metrics")
    monkeypatch.setitem(sys.modules, "opentelemetry", mock_otel)
    monkeypatch.setitem(sys.modules, "opentelemetry.metrics", mock_metrics)
    importlib.reload(otel_mod)
    assert otel_mod._OTEL_SDK_AVAILABLE is True

    monkeypatch.delitem(sys.modules, "opentelemetry.metrics", raising=False)
    monkeypatch.delitem(sys.modules, "opentelemetry", raising=False)
    importlib.reload(otel_mod)
