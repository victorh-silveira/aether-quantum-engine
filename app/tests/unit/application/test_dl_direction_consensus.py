from unittest.mock import patch

from src.application.services.deep_learning.dl_direction_consensus import (
    _candle_bar_direction,
    _dl_vote_weight,
    dl_direction_from_raw,
    resolve_consensus_direction,
)
from src.domain.models.trade import TradeDirection


def test_consensus_keeps_dl_when_raw_strong():
    metrics = {"trade_score": 0.54, "raw_prob": 0.66}
    ctx = {
        "body": -0.002,
        "body_sum_3": -0.006,
        "close_loc": 0.46,
        "variance_ratio": 0.90,
        "ema_spread": -0.001,
        "ret_5": -0.002,
    }
    direction, _ = resolve_consensus_direction(metrics, ctx)
    assert direction == TradeDirection.CALL


def test_consensus_weak_dl_follows_flow():
    metrics = {"trade_score": 0.0, "raw_prob": 0.51, "gate_reason": "conviction"}
    ctx = {
        "body": 0.003,
        "body_sum_3": 0.008,
        "ema_spread": 0.002,
        "ret_5": 0.003,
        "close_loc": 0.62,
        "variance_ratio": 0.95,
        "rel_vol": 0.35,
        "body_streak": 3.0,
        "rsi_slope": 0.02,
    }
    direction, strength = resolve_consensus_direction(metrics, ctx)
    assert direction == TradeDirection.CALL
    assert strength > 0.2


def test_consensus_mean_reversion_put():
    metrics = {"trade_score": 0.52, "raw_prob": 0.52}
    ctx = {
        "body": 0.0,
        "body_sum_3": 0.0,
        "sma_z": 0.005,
        "variance_ratio": 0.75,
        "rsi": 0.5,
        "close_loc": 0.5,
    }
    direction, _ = resolve_consensus_direction(metrics, ctx)
    assert direction == TradeDirection.PUT


def test_consensus_returns_pipeline_dl_when_no_votes():
    direction, strength = resolve_consensus_direction(
        {"raw_prob": 0.49},
        {},
        TradeDirection.PUT,
    )
    assert direction == TradeDirection.PUT
    assert strength == 0.0


def test_consensus_tie_returns_pipeline_dl():
    direction, strength = resolve_consensus_direction({}, {"body": 0.0}, TradeDirection.PUT)
    assert direction == TradeDirection.PUT
    assert strength == 0.0


def test_consensus_uses_dl_dir_when_raw_borderline():
    metrics = {"raw_prob": 0.51}
    ctx = {"sma_z": 0.001}
    direction, _ = resolve_consensus_direction(metrics, ctx, TradeDirection.PUT)
    assert direction == TradeDirection.PUT


def test_consensus_mr_high_variance_ratio():
    metrics = {"trade_score": 0.52, "raw_prob": 0.52}
    ctx = {"sma_z": 0.005, "variance_ratio": 0.90, "body": 0.0, "close_loc": 0.5}
    direction, _ = resolve_consensus_direction(metrics, ctx)
    assert direction == TradeDirection.PUT


def test_dl_vote_weight_branches():
    assert _dl_vote_weight(0.50, 0.62) > _dl_vote_weight(0.50, 0.56)
    assert _dl_vote_weight(0.50, 0.56) > _dl_vote_weight(0.50, 0.54)
    assert _dl_vote_weight(0.50, 0.52) < _dl_vote_weight(0.50, 0.54)


def test_candle_bar_direction_branches():
    assert _candle_bar_direction({}) is None
    assert _candle_bar_direction({"body": 0.002, "close_loc": 0.55}) == TradeDirection.CALL
    assert _candle_bar_direction({"body": -0.002, "close_loc": 0.45}) == TradeDirection.PUT
    assert _candle_bar_direction({"body": 0.002, "close_loc": 0.40}) is None


def test_dl_direction_from_raw_put_and_missing():
    assert dl_direction_from_raw({"raw_prob": 0.44}) == TradeDirection.PUT
    assert dl_direction_from_raw({}) is None


def test_consensus_exact_tie_returns_pipeline_dl():
    with (
        patch(
            "src.application.services.deep_learning.dl_direction_consensus.flow_implied_direction",
            return_value=TradeDirection.CALL,
        ),
        patch(
            "src.application.services.deep_learning.dl_direction_consensus.flow_strength",
            return_value=20 / 28,
        ),
        patch(
            "src.application.services.deep_learning.dl_direction_consensus.binary_direction_vote",
            return_value=TradeDirection.PUT,
        ),
        patch(
            "src.application.services.deep_learning.dl_direction_consensus._candle_bar_direction",
            return_value=None,
        ),
        patch(
            "src.application.services.deep_learning.dl_direction_consensus.sma_extreme_direction",
            return_value=None,
        ),
    ):
        direction, strength = resolve_consensus_direction({}, {"x": 1}, TradeDirection.PUT)
    assert direction == TradeDirection.PUT
    assert strength == 0.0
