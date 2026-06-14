from src.application.services.execution_market_rank import market_decision_score, resolve_market_direction
from src.domain.models.trade import TradeDirection


def test_resolve_market_direction_strong_call():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"raw_prob": 0.88, "execute": True},
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_market_decision_score_higher_for_execute():
    strong = market_decision_score({"raw_prob": 0.80, "val_accuracy": 0.55, "execute": True})
    weak = market_decision_score({"raw_prob": 0.52, "val_accuracy": 0.55, "execute": False})
    assert strong > weak
