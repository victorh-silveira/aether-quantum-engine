from unittest.mock import patch

from src.application.services.execution_direction_fallback import (
    _forced_recovery_pick,
    _last_resort_fallback_pick,
    _scored_fallback_pick,
    build_mandatory_fallback_candidate,
)
from src.domain.models.trade import TradeDirection


def test_scored_fallback_skips_lower_score_candidate():
    decisions = {
        "R_50": {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.70, "raw_prob": 0.70}},
        "R_75": {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.55, "raw_prob": 0.55}},
    }
    picked = _scored_fallback_pick(["R_50", "R_75"], decisions, min_signal=0.45)
    assert picked is not None
    assert picked[0] == "R_50"


def test_last_resort_skips_symbol_when_builders_fail():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "deploy_ok": True},
        },
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.60, "raw_prob": 0.60, "deploy_ok": True},
        },
    }
    with (
        patch(
            "src.application.services.execution_direction_fallback.build_market_execution_candidate",
            side_effect=[None, ("R_75", TradeDirection.CALL, {"trade_score": 0.60})],
        ),
        patch(
            "src.application.services.execution_direction_fallback.build_execution_candidate",
            side_effect=[None, None],
        ),
    ):
        picked = _last_resort_fallback_pick(["R_50", "R_75"], decisions, min_signal=0.0)
    assert picked is not None
    assert picked[0] == "R_75"


def test_last_resort_skips_symbol_without_candidate():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "data", "deploy_ok": True}},
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "deploy_ok": True},
        },
    }
    picked = _last_resort_fallback_pick(["R_50", "R_75"], decisions, min_signal=0.0)
    assert picked is not None
    assert picked[0] == "R_75"


def test_scored_fallback_pick_returns_highest_score():
    decisions = {
        "R_50": {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.55, "raw_prob": 0.58}},
        "R_75": {"direction": TradeDirection.PUT, "metrics": {"gate_reason": "data", "trade_score": 0.70}},
    }
    picked = _scored_fallback_pick(["R_50", "R_75"], decisions, min_signal=0.45)
    assert picked is not None
    assert picked[0] == "R_50"


def test_build_mandatory_fallback_uses_scored_when_forced_recovery_misses():
    with patch(
        "src.application.services.execution_direction_fallback.pick_best_mandatory_candidate",
        return_value=None,
    ):
        best = build_mandatory_fallback_candidate(
            ["R_50"],
            {
                "R_50": {
                    "direction": TradeDirection.CALL,
                    "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "val_accuracy": 0.55},
                }
            },
            recovery_active=True,
            last_loss_symbol="R_10",
            last_loss_direction="PUT",
            min_signal=0.45,
            min_val=0.50,
        )
    assert best is not None
    assert best[1] == TradeDirection.CALL


def test_build_mandatory_fallback_uses_forced_recovery_when_market_rank_empty():
    with patch(
        "src.application.services.execution_direction_fallback.pick_best_mandatory_candidate",
        return_value=None,
    ):
        best = build_mandatory_fallback_candidate(
            ["R_50"],
            {"R_50": {"direction": TradeDirection.PUT, "metrics": {"trade_score": 0.60, "val_accuracy": 0.55}}},
            recovery_active=True,
            last_loss_symbol="R_10",
            last_loss_direction="PUT",
            min_signal=0.45,
            min_val=0.50,
        )
    assert best is not None
    assert best[1] == TradeDirection.PUT


def test_scored_fallback_pick_skips_low_val_accuracy():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "val_accuracy": 0.40},
        },
    }
    picked = _scored_fallback_pick(["R_50"], decisions, min_signal=0.45, min_val=0.50)
    assert picked is None


def test_last_resort_fallback_pick_skips_low_val_accuracy():
    decisions = {
        "R_50": {
            "direction": None,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "val_accuracy": 0.40, "deploy_ok": True},
        },
    }
    picked = _last_resort_fallback_pick(["R_50"], decisions, min_signal=0.45, min_val=0.50)
    assert picked is None


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


def test_scored_fallback_uses_execution_candidate_when_market_build_fails():
    decisions = {
        "R_50": {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.55, "raw_prob": 0.58}},
    }
    with patch(
        "src.application.services.execution_direction_fallback.build_market_execution_candidate",
        return_value=None,
    ):
        picked = _scored_fallback_pick(["R_50"], decisions, min_signal=0.45)
    assert picked is not None
    assert picked[0] == "R_50"


def test_last_resort_returns_candidate_from_execution_builder():
    decisions = {
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.44, "deploy_ok": True},
        },
    }
    with patch(
        "src.application.services.execution_direction_fallback.build_market_execution_candidate",
        return_value=None,
    ):
        picked = _last_resort_fallback_pick(["R_50"], decisions, min_signal=0.0)
    assert picked is not None

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
