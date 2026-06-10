from src.application.services.execution_direction_fallback import (
    _forced_recovery_pick,
    _recovery_metrics_eligible,
    _symbol_priority,
    build_mandatory_fallback_candidate,
)
from src.domain.models.trade import TradeDirection


def test_recovery_metrics_eligible_rejects_low_val_accuracy():
    metrics = {"trade_score": 0.60, "val_accuracy": 0.40}
    assert _recovery_metrics_eligible(metrics, min_signal=0.45, min_val=0.50) is False


def test_recovery_metrics_eligible_rejects_low_trade_score():
    metrics = {"trade_score": 0.40, "val_accuracy": 0.55}
    assert _recovery_metrics_eligible(metrics, min_signal=0.45, min_val=0.50) is False


def test_build_mandatory_fallback_returns_last_resort_at_configured_min_signal():
    decisions = {
        "R_50": {
            "direction": None,
            "metrics": {"trade_score": 0.50},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["R_50"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
        min_signal=0.45,
    )
    assert best is not None
    assert best[1] == TradeDirection.CALL


def test_symbol_priority_recovery_core_only():
    order = _symbol_priority(["R_10", "R_50", "R_75"], "R_50", recovery_core_only=True)
    assert order == ["R_75", "R_50"]


def test_forced_recovery_pick_rejects_low_val_accuracy():
    decisions = {
        "R_75": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.60, "val_accuracy": 0.40, "raw_prob": 0.44},
        },
    }
    assert (
        _forced_recovery_pick(
            ["R_75"],
            decisions,
            TradeDirection.PUT,
            min_signal=0.45,
            min_val=0.50,
        )
        is None
    )


def test_build_mandatory_fallback_returns_none_below_min_signal():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.20, "val_accuracy": 0.50, "raw_prob": 0.55},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["R_50"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
        min_signal=0.45,
    )
    assert best is None
