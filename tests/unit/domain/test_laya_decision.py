"""Testes unitarios para LayaDecision."""

import pytest

from src.domain.models.laya_decision import LayaDecision
from src.domain.models.trade import TradeDirection


def test_laya_decision_allows_call_and_put_only():
    dec = LayaDecision(
        symbol="R_10",
        direction=TradeDirection.CALL,
        probability=0.72,
        prob_call=0.72,
        prob_put=0.28,
        confidence=0.72,
        latency_ms=42.5,
        indicators={"ema_20": 6500.0},
    )
    assert dec.direction is TradeDirection.CALL
    assert dec.probability == 0.72

    with pytest.raises(ValueError, match="CALL ou PUT"):
        LayaDecision(
            symbol="R_10",
            direction=TradeDirection.MULTUP,
            probability=0.6,
            prob_call=0.6,
            prob_put=0.4,
            confidence=0.6,
            latency_ms=10.0,
            indicators={},
        )
