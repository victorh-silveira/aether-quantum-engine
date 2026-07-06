from unittest.mock import patch

from src.application.services.execution_direction_fallback import (
    _forced_recovery_pick,
    _last_resort_fallback_pick,
    _scored_fallback_pick,
    build_mandatory_fallback_candidate,
)
from src.domain.models.trade import TradeDirection
from tests.unit.application.universal_regime_metrics import bear_put_metrics


def test_scored_fallback_skips_lower_score_candidate():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.70, "raw_prob": 0.70, "deploy_ok": True},
        },
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": bear_put_metrics(trade_score=0.55, raw_prob=0.42, calibrated_prob=0.42),
        },
    }
    picked = _scored_fallback_pick(["RDBULL", "RDBEAR"], decisions, min_signal=0.45)
    assert picked is not None
    assert picked[0] == "RDBULL"


def test_last_resort_skips_symbol_when_builders_fail():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "deploy_ok": True},
        },
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": bear_put_metrics(trade_score=0.60, raw_prob=0.42, calibrated_prob=0.42, deploy_ok=True),
        },
    }
    with (
        patch(
            "src.application.services.execution_direction_fallback.build_market_execution_candidate",
            side_effect=[None, ("RDBEAR", TradeDirection.PUT, {"trade_score": 0.60})],
        ),
        patch(
            "src.application.services.execution_direction_fallback.build_execution_candidate",
            side_effect=[None, None],
        ),
    ):
        picked = _last_resort_fallback_pick(["RDBULL", "RDBEAR"], decisions, min_signal=0.0)
    assert picked is not None
    assert picked[0] == "RDBEAR"


def test_last_resort_skips_symbol_without_candidate():
    decisions = {
        "RDBULL": {"direction": None, "metrics": {"gate_reason": "data", "deploy_ok": True}},
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": bear_put_metrics(trade_score=0.55, raw_prob=0.42, calibrated_prob=0.42, deploy_ok=True),
        },
    }
    picked = _last_resort_fallback_pick(["RDBULL", "RDBEAR"], decisions, min_signal=0.0)
    assert picked is not None
    assert picked[0] == "RDBEAR"


def test_scored_fallback_pick_returns_highest_score():
    decisions = {
        "RDBULL": {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.55, "raw_prob": 0.58}},
        "RDBEAR": {"direction": TradeDirection.PUT, "metrics": {"gate_reason": "data", "trade_score": 0.70}},
    }
    picked = _scored_fallback_pick(["RDBULL", "RDBEAR"], decisions, min_signal=0.45)
    assert picked is not None
    assert picked[0] == "RDBULL"


def test_build_mandatory_fallback_uses_scored_when_forced_recovery_misses():
    with patch(
        "src.application.services.execution_direction_fallback.pick_best_mandatory_candidate",
        return_value=None,
    ):
        best = build_mandatory_fallback_candidate(
            ["RDBULL"],
            {
                "RDBULL": {
                    "direction": TradeDirection.CALL,
                    "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "val_accuracy": 0.55},
                }
            },
            recovery_active=True,
            last_loss_symbol="RDBEAR",
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
            ["RDBEAR"],
            {
                "RDBEAR": {
                    "direction": TradeDirection.PUT,
                    "metrics": bear_put_metrics(
                        trade_score=0.60, val_accuracy=0.55, raw_prob=0.42, calibrated_prob=0.42
                    ),
                }
            },
            recovery_active=True,
            last_loss_symbol="RDBEAR",
            last_loss_direction="PUT",
            min_signal=0.45,
            min_val=0.50,
        )
    assert best is not None
    assert best[1] == TradeDirection.PUT


def test_scored_fallback_pick_skips_low_val_accuracy():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "val_accuracy": 0.40},
        },
    }
    picked = _scored_fallback_pick(["RDBULL"], decisions, min_signal=0.45, min_val=0.50)
    assert picked is None


def test_last_resort_fallback_pick_skips_low_val_accuracy():
    decisions = {
        "RDBULL": {
            "direction": None,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "val_accuracy": 0.40, "deploy_ok": True},
        },
    }
    picked = _last_resort_fallback_pick(["RDBULL"], decisions, min_signal=0.45, min_val=0.50)
    assert picked is None


def test_forced_recovery_pick_skips_blocked_symbols():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.70, "val_accuracy": 0.55, "raw_prob": 0.58},
        },
        "RDBEAR": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.55, "val_accuracy": 0.52, "raw_prob": 0.54},
        },
    }
    picked = _forced_recovery_pick(
        ["RDBULL", "RDBEAR"],
        decisions,
        TradeDirection.CALL,
        skip_symbols=frozenset({"RDBULL"}),
        min_signal=0.45,
        min_val=0.50,
    )
    assert picked is not None
    assert picked[0] == "RDBEAR"


def test_scored_fallback_pick_skips_blocked_and_weak_symbols():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.70, "raw_prob": 0.58},
        },
        "RDBEAR": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.20},
        },
    }
    assert (
        _scored_fallback_pick(
            ["RDBULL", "RDBEAR"],
            decisions,
            skip_symbols=frozenset({"RDBULL"}),
            min_signal=0.45,
        )
        is None
    )


def test_scored_fallback_uses_execution_candidate_when_market_build_fails():
    decisions = {
        "RDBULL": {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.55, "raw_prob": 0.58}},
    }
    with patch(
        "src.application.services.execution_direction_fallback.build_market_execution_candidate",
        return_value=None,
    ):
        picked = _scored_fallback_pick(["RDBULL"], decisions, min_signal=0.45)
    assert picked is not None
    assert picked[0] == "RDBULL"


def test_last_resort_returns_candidate_from_execution_builder():
    decisions = {
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": bear_put_metrics(trade_score=0.55, raw_prob=0.42, calibrated_prob=0.42, deploy_ok=True),
        },
    }
    with patch(
        "src.application.services.execution_direction_fallback.build_market_execution_candidate",
        return_value=None,
    ):
        picked = _last_resort_fallback_pick(["RDBEAR"], decisions, min_signal=0.0)
    assert picked is not None

    decisions = {
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": False,
                "trade_score": 0.20,
                "deploy_ok": True,
            },
        },
    }
    assert _last_resort_fallback_pick(["RDBEAR"], decisions, min_signal=0.45) is None


def test_last_resort_fallback_pick_put_side_and_skip():
    decisions = {
        "RDBEAR": {
            "direction": TradeDirection.PUT,
            "metrics": bear_put_metrics(execute=False, trade_score=0.55, raw_prob=0.42, calibrated_prob=0.42),
        },
    }
    picked = _last_resort_fallback_pick(
        ["RDBEAR"],
        decisions,
        skip_symbols=frozenset(),
        min_signal=0.45,
    )
    assert picked is not None
    assert picked[1] == TradeDirection.PUT
    assert _last_resort_fallback_pick(["RDBEAR"], decisions, skip_symbols=frozenset({"RDBEAR"})) is None
