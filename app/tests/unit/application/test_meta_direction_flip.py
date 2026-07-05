import pytest

from src.application.services.meta_direction_flip import (
    META_FLIP_TRADE_SCORE,
    apply_meta_direction_flip,
    flipped_direction,
    should_flip_direction,
)
from src.domain.models.trade import TradeDirection


def test_should_flip_when_payoff_below_threshold_and_meta_applied():
    assert should_flip_direction(TradeDirection.CALL, 0.35, meta_applied=True)
    assert not should_flip_direction(TradeDirection.CALL, 0.55, meta_applied=True)
    assert not should_flip_direction(TradeDirection.PUT, 0.30, meta_applied=False)


def test_flipped_direction_opposes_tcn():
    assert flipped_direction(TradeDirection.CALL) == TradeDirection.PUT
    assert flipped_direction(TradeDirection.PUT) == TradeDirection.CALL


def test_apply_meta_direction_flip_call_to_put():
    metrics: dict = {}
    exec_dir, score = apply_meta_direction_flip(
        TradeDirection.CALL,
        metrics,
        0.38,
        meta_applied=True,
        tcn_probability=0.68,
    )
    assert exec_dir == TradeDirection.PUT
    assert score == pytest.approx(META_FLIP_TRADE_SCORE)
    assert metrics["meta_direction_flip"] is True
    assert metrics["exec_direction"] == "PUT"
    assert metrics["dl_direction"] == "CALL"
    assert metrics["trade_score"] == pytest.approx(META_FLIP_TRADE_SCORE)


def test_apply_meta_direction_flip_put_to_call():
    metrics: dict = {}
    exec_dir, score = apply_meta_direction_flip(
        TradeDirection.PUT,
        metrics,
        0.30,
        meta_applied=True,
        tcn_probability=0.32,
    )
    assert exec_dir == TradeDirection.CALL
    assert metrics["meta_direction_flip"] is True
    assert metrics["direction_inverted"] is True
    assert score == pytest.approx(META_FLIP_TRADE_SCORE)


def test_apply_meta_direction_flip_skips_when_payoff_healthy():
    metrics: dict = {}
    exec_dir, score = apply_meta_direction_flip(
        TradeDirection.CALL,
        metrics,
        0.62,
        meta_applied=True,
        tcn_probability=0.62,
    )
    assert exec_dir == TradeDirection.CALL
    assert score == pytest.approx(0.62)
    assert metrics.get("meta_direction_flip") is not True
