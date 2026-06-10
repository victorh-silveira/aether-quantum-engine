from unittest.mock import patch

from src.application.services.deep_learning.dl_candle_flow import binary_direction_vote
from src.application.services.execution_mandatory_pick import (
    pick_absolute_mandatory_candidate,
    pick_best_mandatory_candidate,
)
from src.application.services.execution_market_rank import (
    build_market_execution_candidate,
    resolve_market_direction,
)
from src.domain.models.trade import TradeDirection


def test_resolve_market_direction_moderate_raw_flips_on_candle_conflict():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "trade_score": 0.55,
            "raw_prob": 0.52,
            "execute": True,
            "binary_ctx": {
                "body": -0.002,
                "body_sum_3": -0.006,
                "close_loc": 0.46,
                "sma_z": 0.001,
                "rsi": 0.5,
                "variance_ratio": 0.90,
                "ema_spread": -0.001,
                "ret_5": -0.002,
            },
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_weak_raw_uses_flow_on_sma_extreme():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "trade_score": 0.55,
            "raw_prob": 0.52,
            "execute": True,
            "binary_ctx": {
                "sma_z": 0.005,
                "variance_ratio": 0.75,
                "body": -0.001,
                "body_sum_3": -0.003,
                "close_loc": 0.46,
                "rsi": 0.5,
            },
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_weak_signal_prefers_candle_over_dl():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "trade_score": 0.0,
            "raw_prob": 0.49,
            "gate_reason": "raw_conviction",
            "binary_ctx": {"body": 0.002, "close_loc": 0.55, "sma_z": 0.0, "rsi": 0.5},
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_resolve_market_direction_weak_raw_inverts_when_candle_disagrees():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "trade_score": 0.52,
            "raw_prob": 0.52,
            "binary_ctx": {"body": -0.002, "close_loc": 0.45, "sma_z": 0.0, "rsi": 0.5},
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_binary_vote_put_on_positive_spread():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "raw_prob": 0.52,
            "binary_ctx": {"rsi": 0.5, "sma_z": 0.0, "close_loc": 0.5, "z_spread": 0.25},
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_binary_vote_call_on_oversold():
    entry = {
        "direction": None,
        "metrics": {
            "raw_prob": 0.51,
            "binary_ctx": {"rsi": 0.40, "sma_z": 0.0, "close_loc": 0.5, "z_spread": 0.0},
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_binary_direction_vote_close_loc_favors_call():
    assert binary_direction_vote({"close_loc": 0.55}) == TradeDirection.CALL


def test_binary_direction_vote_negative_spread_favors_call():
    assert binary_direction_vote({"z_spread": -0.25}) == TradeDirection.CALL


def test_binary_direction_vote_tie_breaks_put_on_high_sma_z():
    ctx = {"sma_z": 0.005, "rsi": 0.40, "close_loc": 0.55, "z_spread": 0.0}
    assert binary_direction_vote(ctx) == TradeDirection.PUT


def test_binary_direction_vote_tie_breaks_call_on_low_sma_z():
    ctx = {"sma_z": -0.005, "rsi": 0.60, "close_loc": 0.45, "z_spread": 0.0}
    assert binary_direction_vote(ctx) == TradeDirection.CALL


def test_resolve_market_direction_uses_flow_when_dl_weak():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "raw_prob": 0.51,
            "binary_ctx": {
                "body": -0.002,
                "body_sum_3": -0.005,
                "sma_z": 0.005,
                "rsi": 0.62,
                "close_loc": 0.42,
                "variance_ratio": 0.75,
                "z_spread": 0.0,
            },
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_sma_extreme_when_binary_vote_tied():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "raw_prob": 0.52,
            "binary_ctx": {"sma_z": 0.005, "variance_ratio": 0.75, "body": 0.0, "close_loc": 0.5},
        },
    }
    with patch(
        "src.application.services.execution_market_rank.flow_implied_direction",
        return_value=None,
    ):
        assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_sma_extreme_call_when_binary_vote_tied():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "raw_prob": 0.52,
            "binary_ctx": {"sma_z": -0.005, "variance_ratio": 0.75, "body": 0.0, "close_loc": 0.5},
        },
    }
    with patch(
        "src.application.services.execution_market_rank.flow_implied_direction",
        return_value=None,
    ):
        assert resolve_market_direction(entry) == TradeDirection.CALL


def test_rank_eligible_skips_when_direction_unresolvable_after_build_fails():
    decisions = {
        "R_50": {
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
            "src.application.services.execution_mandatory_pick.resolve_market_direction",
            return_value=None,
        ),
    ):
        picked = pick_best_mandatory_candidate(
            ["R_50"],
            decisions,
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
        )
    assert picked is None


def test_pick_absolute_skips_when_direction_unresolvable_after_build_fails():
    decisions = {
        "R_50": {
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
            "src.application.services.execution_mandatory_pick.resolve_market_direction",
            return_value=None,
        ),
    ):
        picked = pick_absolute_mandatory_candidate(
            ["R_50"],
            decisions,
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
        )
    assert picked is None


def test_pick_best_uses_forced_candidate_when_market_build_fails():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "deploy_ok": True},
        },
    }
    with patch(
        "src.application.services.execution_mandatory_pick.build_market_execution_candidate",
        return_value=None,
    ):
        picked = pick_best_mandatory_candidate(
            ["R_50"],
            decisions,
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
        )
    assert picked is not None
    assert picked[0] == "R_50"
    assert picked[1] == TradeDirection.CALL


def test_pick_absolute_mandatory_uses_forced_candidate_when_market_build_fails():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.55, "raw_prob": 0.58, "deploy_ok": True},
        },
    }
    with patch(
        "src.application.services.execution_mandatory_pick.build_market_execution_candidate",
        return_value=None,
    ):
        picked = pick_absolute_mandatory_candidate(
            ["R_50"],
            decisions,
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
        )
    assert picked is not None
    assert picked[0] == "R_50"
    assert picked[1] == TradeDirection.CALL


def test_resolve_market_direction_returns_dl_when_ctx_neutral():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"raw_prob": 0.51, "binary_ctx": {"sma_z": 0.001}},
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_build_market_execution_candidate_returns_none_without_direction():
    assert build_market_execution_candidate("R_50", {"direction": None, "metrics": {}}) is None
