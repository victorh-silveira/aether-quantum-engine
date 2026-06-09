from src.application.services.execution_direction import (
    build_mandatory_fallback_candidate,
)
from src.domain.models.trade import TradeDirection


def test_loss_direction_invalid_value():
    best = build_mandatory_fallback_candidate(
        ["R_50"],
        {"R_50": {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.5}}},
        recovery_active=True,
        last_loss_symbol=None,
        last_loss_direction="HOLD",
    )
    assert best is not None
    assert best[0] == "R_50"


def test_build_mandatory_fallback_candidate_non_recovery_uses_raw():
    decisions = {
        "R_50": {
            "direction": None,
            "metrics": {
                "gate_reason": "direction_margin",
                "trade_score": 0.62,
                "raw_prob": 0.44,
            },
        },
    }
    best = build_mandatory_fallback_candidate(
        ["R_50"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is not None
    assert best[1] == TradeDirection.PUT


def test_build_mandatory_fallback_candidate_skips_hard_blocked_symbols():
    decisions = {
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {"gate_reason": "deploy", "trade_score": 0.70},
        },
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"gate_reason": "noise", "trade_score": 0.55},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["R_75", "R_50"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_50",
        last_loss_direction="PUT",
    )
    assert best is not None
    assert best[0] == "R_50"
    assert best[1] == TradeDirection.PUT


def test_build_mandatory_fallback_candidate_recovery_without_loss_direction():
    best = build_mandatory_fallback_candidate(
        ["R_50"],
        {"R_50": {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.6, "raw_prob": 0.55}}},
        recovery_active=True,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is not None
    assert best[1] == TradeDirection.CALL


def test_build_mandatory_fallback_candidate_skips_missing_raw():
    decisions = {
        "R_75": {"direction": None, "metrics": {"trade_score": 0.90}},
        "R_50": {
            "direction": None,
            "metrics": {"trade_score": 0.40, "raw_prob": 0.44},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["R_75", "R_50"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is not None
    assert best[0] == "R_50"


def test_build_mandatory_fallback_candidate_last_resort_skips_missing_entry():
    best = build_mandatory_fallback_candidate(
        ["R_10", "R_50"],
        {"R_50": {"direction": TradeDirection.PUT, "metrics": {"gate_reason": "deploy", "trade_score": 0.5}}},
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is None


def test_build_mandatory_fallback_candidate_last_resort_call_without_recovery():
    best = build_mandatory_fallback_candidate(
        ["R_10", "R_50"],
        {"R_50": {"direction": TradeDirection.PUT, "metrics": {"trade_score": 0.5}}},
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is not None
    assert best[1] == TradeDirection.PUT


def test_build_mandatory_fallback_candidate_last_resort_without_decision():
    best = build_mandatory_fallback_candidate(
        ["R_50"],
        {},
        recovery_active=True,
        last_loss_symbol="R_50",
        last_loss_direction="PUT",
    )
    assert best is not None
    assert best[0] == "R_50"
    assert best[1] == TradeDirection.PUT


def test_build_mandatory_fallback_candidate_recovery_forces_put():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "gate_reason": "cooldown", "trade_score": 0.65},
        },
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "gate_reason": "noise", "trade_score": 0.40},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["R_50", "R_75"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_50",
        last_loss_direction="PUT",
    )
    assert best is not None
    assert best[1] == TradeDirection.PUT
    assert best[0] == "R_75"
