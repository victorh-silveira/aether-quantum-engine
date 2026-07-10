from src.application.services.execution_direction_fallback import (
    _loss_direction,
    build_mandatory_fallback_candidate,
)
from src.domain.models.trade import TradeDirection


def test_loss_direction_call_and_put():
    assert _loss_direction("CALL") == TradeDirection.CALL
    assert _loss_direction("put") == TradeDirection.PUT
    assert _loss_direction(None) is None
    assert _loss_direction("HOLD") is None


def test_loss_direction_invalid_value():
    best = build_mandatory_fallback_candidate(
        ["RDBULL"],
        {
            "RDBULL": {
                "direction": TradeDirection.CALL,
                "metrics": {"trade_score": 0.55, "raw_prob": 0.72, "deploy_ok": True, "val_accuracy": 0.55},
            }
        },
        recovery_active=True,
        last_loss_symbol=None,
        last_loss_direction="HOLD",
    )
    assert best is not None
    assert best[0] == "RDBULL"


def test_build_mandatory_fallback_candidate_non_recovery_uses_raw():
    decisions = {
        "RDBULL": {
            "direction": None,
            "metrics": {
                "gate_reason": "direction_margin",
                "trade_score": 0.62,
                "raw_prob": 0.62,
                "deploy_ok": True,
                "val_accuracy": 0.55,
            },
        },
    }
    best = build_mandatory_fallback_candidate(
        ["RDBULL"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is not None
    assert best[1] == TradeDirection.CALL


def test_build_mandatory_fallback_candidate_skips_hard_blocked_symbols():
    decisions = {
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": {"gate_reason": "data", "trade_score": 0.70, "val_accuracy": 0.55, "raw_prob": 0.30},
        },
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.58, "val_accuracy": 0.52, "raw_prob": 0.72, "deploy_ok": True},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["RDBEAR", "RDBULL"],
        decisions,
        recovery_active=True,
        last_loss_symbol="RDBULL",
        last_loss_direction="PUT",
    )
    assert best is not None
    assert best[0] == "RDBULL"
    assert best[1] == TradeDirection.CALL


def test_build_mandatory_fallback_candidate_recovery_without_loss_direction():
    best = build_mandatory_fallback_candidate(
        ["RDBULL"],
        {
            "RDBULL": {
                "direction": TradeDirection.CALL,
                "metrics": {"trade_score": 0.6, "raw_prob": 0.72, "deploy_ok": True, "val_accuracy": 0.55},
            }
        },
        recovery_active=True,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is not None
    assert best[1] == TradeDirection.CALL


def test_build_mandatory_fallback_candidate_skips_missing_raw():
    decisions = {
        "RDBEAR": {"direction": TradeDirection.PUT, "metrics": {"trade_score": 0.90}},
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.56, "raw_prob": 0.72, "deploy_ok": True, "val_accuracy": 0.55},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["RDBEAR", "RDBULL"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is not None
    assert best[0] == "RDBULL"


def test_build_mandatory_fallback_candidate_last_resort_skips_missing_entry():
    best = build_mandatory_fallback_candidate(
        ["RDBEAR", "RDBULL"],
        {
            "RDBULL": {
                "direction": TradeDirection.CALL,
                "metrics": {"gate_reason": "deploy", "trade_score": 0.55, "raw_prob": 0.72, "deploy_ok": True},
            }
        },
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is not None
    assert best[0] == "RDBULL"


def test_build_mandatory_fallback_candidate_last_resort_call_without_recovery():
    best = build_mandatory_fallback_candidate(
        ["RDBEAR", "RDBULL"],
        {
            "RDBEAR": {
                "direction": TradeDirection.PUT,
                "metrics": {"trade_score": 0.72, "raw_prob": 0.28, "deploy_ok": True, "val_accuracy": 0.55},
            }
        },
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is not None
    assert best[1] == TradeDirection.PUT


def test_build_mandatory_fallback_never_picks_training_symbol():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "gate_reason": "training", "trade_score": 0.80, "raw_prob": 0.70},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["RDBULL"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert best is None


def test_build_mandatory_fallback_candidate_last_resort_without_decision():
    best = build_mandatory_fallback_candidate(
        ["RDBULL"],
        {},
        recovery_active=True,
        last_loss_symbol="RDBULL",
        last_loss_direction="PUT",
    )
    assert best is None


def test_build_mandatory_fallback_recovery_uses_dl_when_loss_direction_missing():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": False,
                "gate_reason": "noise",
                "trade_score": 0.65,
                "raw_prob": 0.65,
                "val_accuracy": 0.55,
                "deploy_ok": True,
            },
        },
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": False,
                "gate_reason": "noise",
                "trade_score": 0.80,
                "raw_prob": 0.20,
                "val_accuracy": 0.62,
                "deploy_ok": True,
                "cross_symbol_features": {"cross_symbol_vol_ratio_diff": -0.1},
            },
        },
    }
    best = build_mandatory_fallback_candidate(
        ["RDBULL", "RDBEAR"],
        decisions,
        recovery_active=True,
        last_loss_symbol="RDBULL",
        last_loss_direction="PUT",
    )
    assert best is not None
    assert best[0] == "RDBEAR"
    assert best[1] == TradeDirection.PUT
    decisions["RDBEAR"]["direction"] = TradeDirection.PUT
    decisions["RDBEAR"]["metrics"]["raw_prob"] = 0.20
    decisions["RDBEAR"]["metrics"]["trade_score"] = 0.80
    best_put = build_mandatory_fallback_candidate(
        ["RDBULL", "RDBEAR"],
        decisions,
        recovery_active=True,
        last_loss_symbol="RDBULL",
        last_loss_direction="CALL",
        min_signal=0.45,
        min_val=0.50,
    )
    assert best_put is not None
    assert best_put[0] == "RDBEAR"
    assert best_put[1] == TradeDirection.PUT


def test_build_mandatory_fallback_skips_blocked_symbol_in_recovery():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": False,
                "trade_score": 0.60,
                "val_accuracy": 0.55,
                "raw_prob": 0.58,
                "deploy_ok": True,
            },
        },
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": False,
                "trade_score": 0.55,
                "val_accuracy": 0.52,
                "raw_prob": 0.42,
                "deploy_ok": True,
            },
        },
    }
    best = build_mandatory_fallback_candidate(
        ["RDBULL", "RDBEAR"],
        decisions,
        recovery_active=True,
        last_loss_symbol="RDBULL",
        last_loss_direction="PUT",
        min_signal=0.45,
        min_val=0.50,
    )
    assert best is not None
