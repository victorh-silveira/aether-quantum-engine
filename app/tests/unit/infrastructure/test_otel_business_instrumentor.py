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


def test_indicator_trade_metrics_exist_only_while_contract_is_active():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.record_indicator_trade(7, "R_10", "CALL", {})
    assert "aether_indicator_active_trade_direction" not in instrumentor.format_prometheus_metrics()
    instrumentor.record_indicator_trade(
        7,
        "R_10",
        "CALL",
        {"indicator_model_version": "v1", "indicator_call_probability": 0.64, "indicator_candle_epoch": 300},
    )
    payload = instrumentor.format_prometheus_metrics()
    assert 'aether_indicator_active_trade_direction{symbol="R_10",contract_id="7"} 1.0' in payload
    assert 'aether_indicator_active_trade_call_probability{symbol="R_10",contract_id="7"} 0.64' in payload
    assert 'aether_indicator_active_trade_candle_epoch{symbol="R_10",contract_id="7"} 300' in payload
    instrumentor.record_trade("R_10", "CALL", contract_id=7, won=True, profit=1.0)
    assert "aether_indicator_active_trade_direction" not in instrumentor.format_prometheus_metrics()


def test_pullback_trade_reports_direction_without_invented_probability():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.record_indicator_trade(
        19,
        "1HZ25V",
        "PUT",
        {"indicator_model_version": "pullback-1-v1", "indicator_candle_epoch": 300},
    )
    payload = instrumentor.format_prometheus_metrics()
    assert 'aether_indicator_active_trade_direction{symbol="1HZ25V",contract_id="19"} -1.0' in payload
    assert 'aether_indicator_active_trade_candle_epoch{symbol="1HZ25V",contract_id="19"} 300' in payload
    assert 'aether_indicator_active_trade_call_probability{symbol="1HZ25V",contract_id="19"}' not in payload


def test_business_metrics_omit_unobserved_session_radar_and_events():
    payload = BusinessMetricsInstrumentor().format_prometheus_metrics()
    assert "aether_session_balance_usd" not in payload
    assert "aether_inference_calibrated" not in payload
    assert "aether_trading_contracts_total" not in payload
    assert "aether_trading_gate_verdicts_total" not in payload
    assert "aether_trading_brier_score" not in payload


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


def test_business_metrics_instrumentor_uses_execution_metric_names():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.record_inference_radar(
        symbol="1HZ75V",
        metrics={"calibrated_prob": 0.61, "predicted_payoff_edge": -0.02},
    )
    payload = instrumentor.format_prometheus_metrics()
    assert 'aether_inference_calibrated{symbol="1HZ75V"} 0.61' in payload
    assert 'aether_inference_payoff_edge{symbol="1HZ75V"} -0.02' in payload


def test_confirmed_contract_uses_its_side_probability_and_direction():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.track_contract_probability(17, 0.72)
    instrumentor.record_trade("1HZ75V", "PUT", contract_id=17, won=False, profit=-10.0)
    payload = instrumentor.format_prometheus_metrics()
    assert 'aether_trading_contracts_total{symbol="1HZ75V",direction="PUT",outcome="LOSS"} 1' in payload
    assert "aether_trading_brier_score 0.5184" in payload
    assert "aether_trading_brier_samples 1" in payload
    assert instrumentor._contract_predictions == {}


def test_unknown_prediction_does_not_create_fabricated_brier():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.record_trade("1HZ75V", "UNKNOWN", contract_id=18, won=True, profit=8.0)
    assert instrumentor.compute_rolling_brier_score() is None
    assert "aether_trading_brier_score" not in instrumentor.format_prometheus_metrics()


def test_radar_uses_raw_probability_and_quoted_edge():
    instrumentor = BusinessMetricsInstrumentor()
    instrumentor.record_inference_radar(
        "1HZ75V",
        direction="CALL",
        metrics={
            "raw_prob": 0.64,
            "calibrated_prob": 0.63,
            "predicted_payoff_edge": -0.3,
            "quote_edge": 0.12,
            "loss_clf_p_loss": 0.42,
        },
    )
    payload = instrumentor.format_prometheus_metrics()
    assert 'aether_inference_prob{symbol="1HZ75V"} 0.64' in payload
    assert 'aether_inference_directional_margin{symbol="1HZ75V"} 0.13' in payload
    assert 'aether_inference_payoff_edge{symbol="1HZ75V"} 0.12' in payload


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


def test_business_metrics_session_uptime_and_stop_win():
    instrumentor = BusinessMetricsInstrumentor()
    assert instrumentor.session_uptime_seconds() == 0.0

    instrumentor.set_session_targets(5000.0, 200.0)
    assert instrumentor.session_uptime_seconds() >= 0.0
    payload = instrumentor.format_prometheus_metrics()
    assert "aether_session_uptime_seconds" in payload
    assert "aether_session_start_time_seconds" in payload
    assert "aether_session_stop_win_reached 0" in payload

    instrumentor.mark_stop_win()
    frozen_uptime = instrumentor.session_uptime_seconds()
    assert frozen_uptime >= 0.0
    assert instrumentor._session_stop_win_time is not None
    payload_stop = instrumentor.format_prometheus_metrics()
    assert "aether_session_stop_win_reached 1" in payload_stop

    instrumentor2 = BusinessMetricsInstrumentor()
    instrumentor2.set_session_targets(1000.0, 50.0)
    assert instrumentor2._session_stop_win_time is None
    instrumentor2.update_balance(1055.0)
    assert instrumentor2._session_stop_win_time is not None
