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


def test_business_metrics_omit_unobserved_session_radar_and_events():
    payload = BusinessMetricsInstrumentor().format_prometheus_metrics()
    assert "aether_session_balance_usd" not in payload
    assert "aether_inference_calibrated" not in payload
    assert "aether_trading_contracts_total" not in payload
    assert "aether_trading_gate_verdicts_total" not in payload


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


def test_business_metrics_instrumentor_session_targets_and_financials():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.set_session_targets(100.0, 4.31)
    instrumentor.update_balance(102.15)
    instrumentor.update_active_contracts_count(2)

    payload = instrumentor.format_prometheus_metrics()
    assert "aether_session_start_balance_usd 100.0" in payload
    assert "aether_session_target_win_usd 4.31" in payload
    assert "aether_session_balance_usd 102.15" in payload
    assert "aether_session_target_balance_usd 104.31" in payload
    assert "aether_session_profit_usd 2.15" in payload
    assert "aether_session_remaining_usd 2.16" in payload
    assert "aether_session_active_trades 2" in payload
    assert "aether_session_progress_pct" in payload
    assert "aether_session_roi_pct" in payload


def test_business_metrics_instrumentor_inference_radar():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.record_inference_radar(
        symbol="1HZ75V",
        direction="CALL",
        stake=1.50,
        metrics={
            "prob": 0.62,
            "calibrated_prob": 0.58,
            "directional_margin": 0.08,
            "payoff_edge": 0.045,
            "conviction": 0.70,
            "p_loss": 0.35,
            "p_eff": 0.65,
            "loss_clf_flip": True,
            "anti_trend_lock_active": False,
        },
    )

    payload = instrumentor.format_prometheus_metrics()
    assert 'aether_trading_direction{symbol="1HZ75V"} 1' in payload
    assert 'aether_trading_stake_usd{symbol="1HZ75V"} 1.5' in payload
    assert 'aether_inference_prob{symbol="1HZ75V"} 0.62' in payload
    assert 'aether_inference_calibrated{symbol="1HZ75V"} 0.58' in payload
    assert 'aether_inference_directional_margin{symbol="1HZ75V"} 0.08' in payload
    assert 'aether_inference_payoff_edge{symbol="1HZ75V"} 0.045' in payload
    assert 'aether_inference_conviction{symbol="1HZ75V"} 0.7' in payload
    assert 'aether_loss_classifier_p_loss{symbol="1HZ75V"} 0.35' in payload
    assert 'aether_loss_classifier_p_eff{symbol="1HZ75V"} 0.65' in payload
    assert 'aether_loss_classifier_flip_active{symbol="1HZ75V"} 1' in payload
    assert 'aether_anti_trend_lock_active{symbol="1HZ75V"} 0' in payload


def test_business_metrics_instrumentor_radar_put_and_empty_metrics():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.record_inference_radar(
        symbol="1HZ75V",
        direction="PUT",
        stake=2.0,
        metrics={},
    )
    payload = instrumentor.format_prometheus_metrics()
    assert 'aether_trading_direction{symbol="1HZ75V"} -1' in payload
    assert 'aether_trading_stake_usd{symbol="1HZ75V"} 2.0' in payload
    assert 'aether_inference_calibrated{symbol="1HZ75V"}' not in payload
    assert 'aether_inference_payoff_edge{symbol="1HZ75V"}' not in payload
    assert 'aether_loss_classifier_p_eff{symbol="1HZ75V"}' not in payload


def test_business_metrics_instrumentor_uses_execution_metric_names():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.record_inference_radar(
        symbol="1HZ75V",
        metrics={"calibrated_prob": 0.61, "predicted_payoff_edge": -0.02, "loss_clf_p_eff": 0.72},
    )
    payload = instrumentor.format_prometheus_metrics()
    assert 'aether_inference_calibrated{symbol="1HZ75V"} 0.61' in payload
    assert 'aether_inference_payoff_edge{symbol="1HZ75V"} -0.02' in payload
    assert 'aether_loss_classifier_p_eff{symbol="1HZ75V"} 0.72' in payload


def test_business_metrics_instrumentor_flat_and_trade_variants():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.record_inference_radar(
        symbol="1HZ75V",
        direction="FLAT",
        stake_usd=0.0,
    )
    instrumentor.record_trade(won=False, profit=-10.0, stake=10.0)
    payload = instrumentor.format_prometheus_metrics()
    assert 'aether_trading_direction{symbol="1HZ75V"} 0.0' in payload
    assert 'aether_trading_contracts_total{symbol="1HZ75V",direction="CALL",outcome="LOSS"} 1' in payload
