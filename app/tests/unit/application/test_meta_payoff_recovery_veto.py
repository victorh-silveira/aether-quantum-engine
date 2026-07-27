from src.application.services.meta_payoff_veto_gate import (
    _recovery_active,
    is_execution_signal_vetoed,
    should_veto_meta_payoff_negative_zscore,
)
from src.domain.models.trade import TradeDirection
from tests.unit.application.test_meta_payoff_veto_gate import _risk_manager, _stamp_negative_zscore


def test_recovery_mild_severe_zscore_stays_soft_with_negative_edge():
    metrics = {
        "predicted_payoff_edge": -0.05,
        "edge_expectancy": "LOSS_EXPECTED",
        "trade_score": 0.70,
    }
    _stamp_negative_zscore(metrics, z_score=-2.10)
    rm = _risk_manager(linear=1, pending=20.0)
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL, risk_manager=rm) is False
    assert metrics["meta_veto_mode"] == "soft"
    assert metrics.get("meta_recovery_severe_z_soft") is True
    assert metrics.get("meta_recovery_severe_z_veto") is not True


def test_recovery_catastrophic_zscore_triggers_soft_veto():
    metrics = {
        "predicted_payoff_edge": -0.30,
        "edge_expectancy": "LOSS_EXPECTED",
        "trade_score": 0.70,
    }
    _stamp_negative_zscore(metrics, z_score=-2.90)
    rm = _risk_manager(linear=1, pending=20.0)
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL, risk_manager=rm) is False
    assert metrics["meta_veto_mode"] == "soft"
    assert metrics.get("meta_recovery_severe_z_veto") is True
    assert metrics.get("meta_recovery_severe_z_soft") is True
    assert is_execution_signal_vetoed(metrics) is False


def test_recovery_severe_positive_edge_stays_soft():
    metrics = {
        "predicted_payoff_edge": 0.08,
        "edge_expectancy": "NO_EDGE_NEUTRAL",
        "trade_score": 0.70,
    }
    _stamp_negative_zscore(metrics, z_score=-2.10)
    rm = _risk_manager(linear=2, pending=40.0)
    assert (
        should_veto_meta_payoff_negative_zscore(
            metrics,
            direction=TradeDirection.PUT,
            risk_manager=rm,
            recovery_active=True,
        )
        is False
    )
    assert metrics["meta_veto_mode"] == "soft"
    assert metrics.get("meta_recovery_severe_z_soft") is True
    assert metrics.get("meta_recovery_severe_z_veto") is not True


def test_severe_zscore_without_recovery_stays_soft():
    metrics = {
        "predicted_payoff_edge": -0.05,
        "edge_expectancy": "LOSS_EXPECTED",
        "trade_score": 0.70,
    }
    _stamp_negative_zscore(metrics, z_score=-2.10)
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL) is False
    assert metrics["meta_veto_mode"] == "soft"
    assert metrics.get("meta_recovery_severe_z") is False


def test_recovery_active_via_pending_only_stays_soft_below_catastrophic():
    assert _recovery_active(None) is False
    assert _recovery_active(type("RM", (), {"consecutive_losses_linear": 0})()) is False
    rm_fn = type(
        "RM",
        (),
        {
            "consecutive_losses_linear": 0,
            "pending_loss_total": lambda self: 12.5,
            "pending_loss": {},
        },
    )()
    assert _recovery_active(rm_fn) is True
    rm_map = type(
        "RM",
        (),
        {"consecutive_losses_linear": 0, "pending_loss": {"R_10": 8.0}},
    )()
    assert _recovery_active(rm_map) is True
    metrics = {
        "predicted_payoff_edge": -0.08,
        "edge_expectancy": "LOSS_EXPECTED",
        "trade_score": 0.66,
    }
    _stamp_negative_zscore(metrics, z_score=-2.10)
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.PUT, risk_manager=rm_map) is False
    assert metrics["meta_veto_mode"] == "soft"
    assert metrics.get("meta_recovery_severe_z_soft") is True
    assert metrics.get("meta_recovery_severe_z_veto") is not True
