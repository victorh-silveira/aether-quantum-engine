import pytest

from src.application.services.meta_direction_flip import (
    apply_configured_direction_invert,
    invert_execution_direction_enabled,
)
from src.domain.models.trade import TradeDirection


def test_invert_execution_direction_enabled_reads_flag():
    assert invert_execution_direction_enabled(None) is False
    assert invert_execution_direction_enabled({}) is False
    assert invert_execution_direction_enabled({"invert_execution_direction": True}) is True


def test_apply_configured_direction_invert_mirrors_prob_and_side():
    cal, raw, direction, inverted = apply_configured_direction_invert(
        0.40,
        0.42,
        TradeDirection.PUT,
        exec_cfg={"invert_execution_direction": True},
    )
    assert inverted is True
    assert cal == pytest.approx(0.60)
    assert raw == pytest.approx(0.58)
    assert direction == TradeDirection.CALL


def test_apply_configured_direction_invert_noop_when_disabled():
    cal, raw, direction, inverted = apply_configured_direction_invert(
        0.40,
        0.42,
        TradeDirection.PUT,
        exec_cfg={"invert_execution_direction": False},
    )
    assert inverted is False
    assert cal == pytest.approx(0.40)
    assert raw == pytest.approx(0.42)
    assert direction == TradeDirection.PUT


def test_apply_configured_direction_invert_infers_when_direction_missing():
    cal, raw, direction, inverted = apply_configured_direction_invert(
        0.40,
        0.40,
        None,
        exec_cfg={"invert_execution_direction": True},
    )
    assert inverted is True
    assert cal == pytest.approx(0.60)
    assert direction == TradeDirection.CALL
