from unittest.mock import patch

from src.application.services.execution_direction_fallback import (
    _scored_fallback_pick,
    build_mandatory_fallback_candidate,
)
from src.domain.models.trade import TradeDirection


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
            {"R_50": {"direction": TradeDirection.CALL, "metrics": {"trade_score": 0.55, "raw_prob": 0.58}}},
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
