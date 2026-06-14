from src.application.services.deep_learning.dl_gating import (
    direction_from_raw_prob,
    gating_block_reason,
    should_execute,
)
from src.domain.models.trade import TradeDirection


def test_confidence_threshold_call():
    assert direction_from_raw_prob(0.80) == TradeDirection.CALL


def test_confidence_threshold_put():
    assert direction_from_raw_prob(0.20) == TradeDirection.PUT


def test_confidence_abstains_near_half():
    assert direction_from_raw_prob(0.52) is None


def test_gating_blocks_low_val_accuracy():
    assert gating_block_reason(0.80, 0.50, min_val_accuracy=0.53) == "val_acc"


def test_gating_allows_strong_signal():
    assert should_execute(0.80, 0.55, min_val_accuracy=0.53) is True
