import pytest

from src.application.services.bb_width_adaptive_squeeze import (
    record_bb_width,
    reset_bb_width_buffer,
)
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings
from src.application.services.execution_runtime_config import resolve_meta_payoff_veto_config
from src.application.services.meta_payoff_regression import (
    apply_meta_regression_edge,
    calibration_neutral_axis_drift,
)
from src.application.services.regime_micro_freeze import micro_volatility_squeeze_active
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_bb_buffer():
    reset_bb_width_buffer()
    yield
    reset_bb_width_buffer()


def _prime_bb(value: float, count: int | None = None) -> None:
    window = (
        count
        if count is not None
        else int(load_indicator_config_from_settings()["windows"]["bb_width_harmonic_window"])
    )
    for _ in range(window):
        record_bb_width(value)


def test_calibration_neutral_axis_drift_branches():
    assert calibration_neutral_axis_drift(None, 0.5) is False
    assert calibration_neutral_axis_drift(0.5, None) is False
    assert calibration_neutral_axis_drift(0.6, 0.4) is True
    assert calibration_neutral_axis_drift(0.4, 0.6) is True
    assert calibration_neutral_axis_drift(0.6, 0.7) is False
    assert calibration_neutral_axis_drift(0.4, 0.3) is False


def test_micro_volatility_squeeze_active_bb_width():
    _prime_bb(0.050)
    metrics = {"indicators": {"bb_width": 0.015}}
    assert micro_volatility_squeeze_active(metrics) is True


def test_micro_volatility_squeeze_active_tick_deceleration():
    metrics = {"flow_features": {"micro_tick_acceleration": -0.01}}
    assert micro_volatility_squeeze_active(metrics) is True


def test_apply_meta_regression_edge_positive_keeps_organic_score():
    metrics = {}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        0.12,
        meta_applied=True,
        base_score=0.74,
    )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(0.74)
    assert metrics["predicted_payoff_edge"] == pytest.approx(0.12)


def test_apply_meta_regression_edge_loss_expected_keeps_direction():
    metrics = {
        "indicators": {"bb_width": 0.12},
        "flow_features": {"micro_tick_acceleration": 0.03},
    }
    direction, score = apply_meta_regression_edge(
        TradeDirection.PUT,
        metrics,
        -0.25,
        meta_applied=True,
        base_score=0.68,
        symbol="R_10",
    )
    assert direction == TradeDirection.PUT
    assert score == pytest.approx(0.68)
    assert metrics.get("meta_negative_edge") is True
    assert metrics.get("meta_soft_kelly") is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(
        float(resolve_meta_payoff_veto_config()["soft_veto_score_factor"])
    )
    assert metrics["gate_verdict"] == "SOFT_SIZE"
    assert metrics["gate_verdict_reason"] == "meta_negative_edge"
    assert metrics.get("meta_squeeze_downgrade") is not True


def test_apply_meta_regression_edge_bb_compression_triggers_squeeze_even_positive_edge(caplog):
    _prime_bb(0.060)
    metrics = {
        "indicators": {"bb_width": 0.010},
        "flow_features": {"micro_tick_acceleration": 0.05},
    }
    with caplog.at_level("INFO"):
        direction, score = apply_meta_regression_edge(
            TradeDirection.CALL,
            metrics,
            0.18,
            meta_applied=True,
            base_score=0.72,
            symbol="R_10",
        )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(float(resolve_meta_payoff_veto_config()["squeeze_trade_score"]))
    assert metrics["meta_squeeze_downgrade"] is True
    assert any("[D-SQUEEZE]" in record.message for record in caplog.records)


def test_apply_meta_regression_edge_mild_negative_keeps_call():
    metrics = {"indicators": {"bb_width": 0.12}, "flow_features": {"micro_tick_acceleration": 0.03}}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        -0.05,
        meta_applied=True,
        base_score=0.71,
    )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(0.71)
    assert metrics.get("meta_negative_edge") is True
    assert metrics.get("meta_soft_kelly") is True
    assert metrics.get("meta_squeeze_downgrade") is not True


def test_apply_meta_regression_edge_not_applied_uses_base_score():
    metrics = {"meta_negative_edge": True, "meta_soft_kelly": True, "kelly_fraction_scale": 0.85}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        -0.30,
        meta_applied=False,
        base_score=0.66,
    )
    assert score == pytest.approx(0.66)
    assert metrics.get("meta_squeeze_downgrade") is not True
    assert metrics.get("meta_soft_kelly") is None
    assert metrics.get("meta_negative_edge") is None
    assert "predicted_payoff_edge" not in metrics


def test_apply_meta_regression_edge_positive_clears_soft_kelly():
    metrics = {"kelly_fraction_scale": 1.0, "meta_soft_kelly": True, "meta_negative_edge": True}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        0.12,
        meta_applied=True,
        base_score=0.74,
    )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(0.74)
    assert metrics.get("meta_soft_kelly") is None
    assert metrics.get("meta_negative_edge") is None
    assert metrics["kelly_fraction_scale"] == pytest.approx(1.0)


def test_apply_meta_regression_edge_negative_compounds_existing_scale():
    metrics = {"kelly_fraction_scale": 0.80}
    apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        -1.0,
        meta_applied=True,
        base_score=0.55,
    )
    factor = float(resolve_meta_payoff_veto_config()["soft_veto_strong_factor"])
    assert metrics["kelly_fraction_scale"] == pytest.approx(max(0.05, 0.80 * factor))
    assert metrics["meta_soft_kelly"] is True
    assert metrics["meta_soft_strong"] is True


def test_apply_meta_regression_edge_ignores_calibration_neutral_drift():
    metrics = {"raw_prob": 0.43, "calibrated_prob": 0.54}
    direction, score = apply_meta_regression_edge(
        TradeDirection.PUT,
        metrics,
        0.12,
        meta_applied=True,
        base_score=0.54,
    )
    assert direction == TradeDirection.PUT
    assert score == pytest.approx(0.54)
    assert metrics.get("gate_reason") != "calibration_neutral_drift"


def test_meta_squeeze_trade_score_constant():
    assert pytest.approx(0.52) == float(resolve_meta_payoff_veto_config()["squeeze_trade_score"])


def test_apply_meta_regression_edge_strong_negative_uses_strong_factor():
    metrics = {"kelly_fraction_scale": 1.0}
    apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        -1.0,
        meta_applied=True,
        base_score=0.62,
    )
    cfg = resolve_meta_payoff_veto_config()
    assert metrics["meta_soft_strong"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(float(cfg["soft_veto_strong_factor"]))
    assert metrics["gate_verdict_reason"] == "meta_negative_edge"


def test_apply_meta_mild_negative_uses_mild_factor():
    metrics = {"kelly_fraction_scale": 1.0}
    apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        -0.25,
        meta_applied=True,
        base_score=0.62,
    )
    cfg = resolve_meta_payoff_veto_config()
    assert metrics.get("meta_soft_strong") is False
    assert metrics["kelly_fraction_scale"] == pytest.approx(float(cfg["soft_veto_score_factor"]))


def test_apply_meta_sat_vs_neg_cal_soft_kelly():
    metrics = {"cal_side_edge": -0.029, "kelly_fraction_scale": 1.0}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        0.85,
        meta_applied=True,
        base_score=0.52,
    )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(0.52)
    assert metrics.get("meta_sat_soft") is True
    assert metrics.get("meta_soft_kelly") is True
    assert metrics.get("meta_negative_edge") is None
    assert metrics["gate_verdict"] == "SOFT_SIZE"
    assert metrics["gate_verdict_reason"] == "meta_sat_vs_neg_cal"
    assert metrics["kelly_fraction_scale"] == pytest.approx(
        float(resolve_meta_payoff_veto_config()["soft_veto_score_factor"])
    )


def test_apply_meta_sat_with_positive_cal_edge_no_soft():
    metrics = {"cal_side_edge": 0.015, "kelly_fraction_scale": 1.0}
    direction, score = apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        0.85,
        meta_applied=True,
        base_score=0.55,
    )
    assert direction == TradeDirection.CALL
    assert score == pytest.approx(0.55)
    assert metrics.get("meta_sat_soft") is None
    assert metrics.get("meta_soft_kelly") is None
    assert metrics["kelly_fraction_scale"] == pytest.approx(1.0)


def test_apply_meta_sat_without_cal_side_edge_no_soft():
    metrics = {"kelly_fraction_scale": 1.0}
    apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        0.85,
        meta_applied=True,
        base_score=0.55,
    )
    assert metrics.get("meta_sat_soft") is None
    assert metrics.get("meta_soft_kelly") is None


def test_apply_meta_sat_ignores_bad_cal_side_edge():
    metrics = {"cal_side_edge": object(), "kelly_fraction_scale": 1.0}
    apply_meta_regression_edge(
        TradeDirection.CALL,
        metrics,
        0.85,
        meta_applied=True,
        base_score=0.55,
    )
    assert metrics.get("meta_sat_soft") is None
