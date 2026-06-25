from src.application.services.execution_direction_fallback import (
    _forced_recovery_pick,
    _last_resort_fallback_pick,
    _loss_direction,
    _scored_fallback_pick,
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
            "direction": TradeDirection.PUT,
            "metrics": {"gate_reason": "data", "trade_score": 0.70, "val_accuracy": 0.55},
        },
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.58, "val_accuracy": 0.52},
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
    assert best is not None
    assert best[0] == "R_50"


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


def test_build_mandatory_fallback_never_picks_training_symbol():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "gate_reason": "training", "trade_score": 0.80, "raw_prob": 0.70},
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


def test_build_mandatory_fallback_candidate_last_resort_without_decision():
    best = build_mandatory_fallback_candidate(
        ["R_50"],
        {},
        recovery_active=True,
        last_loss_symbol="R_50",
        last_loss_direction="PUT",
    )
    assert best is None


def test_build_mandatory_fallback_recovery_uses_dl_when_loss_direction_missing():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "gate_reason": "trend_conflict", "trade_score": 0.65, "val_accuracy": 0.55},
        },
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "gate_reason": "noise", "trade_score": 0.58, "val_accuracy": 0.52},
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
    assert best[1] == TradeDirection.CALL
    decisions["R_75"]["direction"] = TradeDirection.PUT
    best_put = build_mandatory_fallback_candidate(
        ["R_50", "R_75"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_50",
        last_loss_direction="PUT",
        min_signal=0.45,
        min_val=0.50,
    )
    assert best_put is not None
    assert best_put[1] == TradeDirection.PUT
    assert best_put[0] == "R_75"


def test_build_mandatory_fallback_skips_blocked_symbol_in_recovery():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.60, "val_accuracy": 0.55, "raw_prob": 0.58},
        },
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.55, "val_accuracy": 0.52, "raw_prob": 0.54},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["R_50", "R_75"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_50",
        last_loss_direction="CALL",
        skip_symbols=frozenset({"R_50"}),
        min_signal=0.45,
        min_val=0.50,
    )
    assert best is not None
    assert best[0] == "R_75"


def test_forced_recovery_pick_skips_blocked_symbols():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.70, "val_accuracy": 0.55, "raw_prob": 0.58},
        },
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.55, "val_accuracy": 0.52, "raw_prob": 0.54},
        },
    }
    picked = _forced_recovery_pick(
        ["R_50", "R_75"],
        decisions,
        TradeDirection.CALL,
        skip_symbols=frozenset({"R_50"}),
        min_signal=0.45,
        min_val=0.50,
    )
    assert picked is not None
    assert picked[0] == "R_75"


def test_scored_fallback_pick_skips_blocked_and_weak_symbols():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.70, "raw_prob": 0.58},
        },
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.20},
        },
    }
    assert (
        _scored_fallback_pick(
            ["R_50", "R_75"],
            decisions,
            skip_symbols=frozenset({"R_50"}),
            min_signal=0.45,
        )
        is None
    )


def test_last_resort_fallback_pick_skips_below_min_signal():
    decisions = {
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": False, "trade_score": 0.20},
        },
    }
    assert _last_resort_fallback_pick(["R_50"], decisions, min_signal=0.45) is None


def test_last_resort_fallback_pick_put_side_and_skip():
    decisions = {
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": False, "trade_score": 0.55, "raw_prob": 0.44},
        },
    }
    picked = _last_resort_fallback_pick(
        ["R_50"],
        decisions,
        skip_symbols=frozenset(),
        min_signal=0.45,
    )
    assert picked is not None
    assert picked[1] == TradeDirection.PUT
    assert _last_resort_fallback_pick(["R_50"], decisions, skip_symbols=frozenset({"R_50"})) is None
