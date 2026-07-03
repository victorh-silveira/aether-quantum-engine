from src.application.services.execution_direction_micro_boundary import validate_micro_boundary_exhaustion
from src.domain.models.trade import TradeDirection


def test_call_top_keltner_downgrades_score():
    metrics = {"trade_score": 0.88, "indicators": {"keltner": 1.16, "bb_pct_b": 0.50}}
    assert validate_micro_boundary_exhaustion(TradeDirection.CALL, metrics) is True
    assert metrics["trade_score"] == 0.55
    assert metrics["micro_boundary_exhaustion"] is True
    assert metrics["micro_boundary_side"] == "upper"


def test_call_top_bollinger_saturation_downgrades_score():
    metrics = {"trade_score": 0.90, "indicators": {"keltner": 0.60, "bb_pct_b": 0.97}}
    assert validate_micro_boundary_exhaustion(TradeDirection.CALL, metrics) is True
    assert metrics["trade_score"] == 0.55


def test_put_bottom_keltner_downgrades_score():
    metrics = {"trade_score": 0.84, "indicators": {"keltner": -0.15, "bb_pct_b": 0.50}}
    assert validate_micro_boundary_exhaustion(TradeDirection.PUT, metrics) is True
    assert metrics["trade_score"] == 0.55
    assert metrics["micro_boundary_side"] == "lower"


def test_put_bottom_bollinger_saturation_downgrades_score():
    metrics = {"trade_score": 0.70, "indicators": {"keltner": 0.40, "bb_pct_b": 0.03}}
    assert validate_micro_boundary_exhaustion(TradeDirection.PUT, metrics) is True
    assert metrics["trade_score"] == 0.55


def test_no_downgrade_when_price_in_mid_channel():
    metrics = {"trade_score": 0.82, "indicators": {"keltner": 0.55, "bb_pct_b": 0.50}}
    assert validate_micro_boundary_exhaustion(TradeDirection.CALL, metrics) is False
    assert metrics["trade_score"] == 0.82
    assert "micro_boundary_exhaustion" not in metrics


def test_downgrade_never_inflates_already_low_score():
    metrics = {"trade_score": 0.40, "indicators": {"keltner": 1.20}}
    assert validate_micro_boundary_exhaustion(TradeDirection.CALL, metrics) is True
    assert metrics["trade_score"] == 0.40
