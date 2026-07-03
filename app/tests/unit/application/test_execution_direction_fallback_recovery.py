from unittest.mock import patch

from src.application.services.execution_direction_fallback import (
    _forced_recovery_pick,
    _last_resort_fallback_pick,
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
        "RDBULL": {
            "direction": None,
            "metrics": {"trade_score": 0.50, "raw_prob": 0.50},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["RDBULL"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
        min_signal=0.45,
    )
    assert best is not None
    assert best[1] == TradeDirection.PUT


def test_symbol_priority_recovery_core_only():
    order = _symbol_priority(["RDBEAR", "RDBULL"], "RDBULL", recovery_core_only=True)
    assert order == ["RDBEAR", "RDBULL"]


def test_symbol_priority_reuses_core_when_tail_empty():
    order = _symbol_priority(["RDBULL", "RDBEAR"], "RDBULL", recovery_core_only=False)
    assert order == ["RDBEAR", "RDBULL"]


def test_last_resort_fallback_uses_raw_when_market_direction_missing():
    decisions = {
        "RDBULL": {
            "direction": None,
            "metrics": {"trade_score": 0.50, "raw_prob": 0.44, "deploy_ok": True},
        },
    }
    with patch(
        "src.application.services.execution_direction_fallback.build_execution_candidate",
        return_value=None,
    ):
        picked = _last_resort_fallback_pick(["RDBULL"], decisions, min_signal=0.0)
    assert picked is not None
    assert picked[1] == TradeDirection.PUT


def test_forced_recovery_pick_skips_gate_blocked():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"deploy_ok": False, "trade_score": 0.60, "val_accuracy": 0.55},
        },
    }
    assert _forced_recovery_pick(["RDBULL"], decisions, TradeDirection.CALL) is None


def test_forced_recovery_pick_rejects_low_val_accuracy():
    decisions = {
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.60, "val_accuracy": 0.40, "raw_prob": 0.44},
        },
    }
    assert (
        _forced_recovery_pick(
            ["RDBEAR"],
            decisions,
            TradeDirection.PUT,
            min_signal=0.45,
            min_val=0.50,
        )
        is None
    )


def test_build_mandatory_fallback_last_resort_below_min_signal():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": False,
                "trade_score": 0.20,
                "val_accuracy": 0.50,
                "deploy_ok": True,
            },
        },
    }
    best = build_mandatory_fallback_candidate(
        ["RDBULL"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
        min_signal=0.45,
    )
    assert best is None
