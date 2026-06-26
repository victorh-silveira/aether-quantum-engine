import pytest

from src.application.services.deep_learning.dl_gating import (
    direction_from_raw_prob,
    resolve_confidence_thresholds,
    resolve_edge,
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


def test_direction_from_raw_prob_call():
    assert direction_from_raw_prob(0.80) == TradeDirection.CALL


def test_direction_from_raw_prob_put():
    assert direction_from_raw_prob(0.20) == TradeDirection.PUT


def test_direction_from_raw_prob_gray_zone():
    assert direction_from_raw_prob(0.52, call_threshold=0.75, put_threshold=0.25) is None
