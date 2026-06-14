import pytest

from src.application.services.deep_learning.dl_gating import (
    direction_from_raw_prob,
    gating_block_reason,
    resolve_confidence_thresholds,
    resolve_edge,
    should_execute,
)
from src.domain.models.trade import TradeDirection


def test_resolve_edge():
    assert resolve_edge(0.80) == pytest.approx(0.30)
    assert resolve_edge(0.50) == 0.0


def test_confidence_thresholds_from_params():
    call_thr, put_thr = resolve_confidence_thresholds(
        {"confidence_call_threshold": 0.80, "confidence_put_threshold": 0.20}
    )
    assert call_thr == 0.80
    assert put_thr == 0.20


def test_should_execute_strong_call():
    assert should_execute(0.80, 0.55, min_val_accuracy=0.53) is True


def test_gating_blocks_weak_signal():
    assert gating_block_reason(0.52, 0.55) == "confidence"


def test_gating_blocks_low_val_accuracy():
    assert gating_block_reason(0.80, 0.50, min_val_accuracy=0.53) == "val_acc"


def test_direction_from_raw_prob_put():
    assert direction_from_raw_prob(0.20) == TradeDirection.PUT
