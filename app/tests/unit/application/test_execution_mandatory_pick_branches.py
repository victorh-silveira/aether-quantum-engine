from unittest.mock import patch

from src.application.services.execution_mandatory_pick import (
    pick_absolute_mandatory_candidate,
    pick_best_mandatory_candidate,
)
from src.application.services.execution_market_rank import build_market_execution_candidate
from src.domain.models.trade import TradeDirection


def test_rank_eligible_skips_when_direction_unresolvable_after_build_fails():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "deploy_ok": True},
        },
    }
    with (
        patch(
            "src.application.services.execution_mandatory_pick.build_market_execution_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_mandatory_pick.mandatory_pool_eligible",
            return_value=True,
        ),
        patch(
            "src.application.services.execution_mandatory_pick.build_execution_candidate",
            return_value=None,
        ),
    ):
        picked = pick_best_mandatory_candidate(
            ["R_10"],
            decisions,
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
        )
    assert picked is None


def test_pick_absolute_skips_when_direction_unresolvable_after_build_fails():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "deploy_ok": True},
        },
    }
    with (
        patch(
            "src.application.services.execution_mandatory_pick.build_market_execution_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_mandatory_pick.mandatory_pool_eligible",
            return_value=True,
        ),
        patch(
            "src.application.services.execution_mandatory_pick.build_execution_candidate",
            return_value=None,
        ),
    ):
        picked = pick_absolute_mandatory_candidate(
            ["R_10"],
            decisions,
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
        )
    assert picked is None


def test_pick_best_uses_forced_candidate_when_market_build_fails():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "deploy_ok": True, "val_accuracy": 0.60},
        },
    }
    with patch(
        "src.application.services.execution_mandatory_pick.build_market_execution_candidate",
        return_value=None,
    ):
        picked = pick_best_mandatory_candidate(
            ["R_10"],
            decisions,
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
        )
    assert picked is not None
    assert picked[0] in {"R_10", "R_50"}
    assert picked[1] == TradeDirection.CALL


def test_pick_absolute_mandatory_uses_forced_candidate_when_market_build_fails():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "deploy_ok": True, "val_accuracy": 0.60},
        },
    }
    with patch(
        "src.application.services.execution_mandatory_pick.build_market_execution_candidate",
        return_value=None,
    ):
        picked = pick_absolute_mandatory_candidate(
            ["R_10"],
            decisions,
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
        )
    assert picked is not None
    assert picked[0] in {"R_10", "R_50"}
    assert picked[1] == TradeDirection.CALL


def test_build_market_execution_candidate_returns_none_without_direction():
    assert build_market_execution_candidate("R_10", {"direction": None, "metrics": {}}) is None
