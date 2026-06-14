from src.application.services.execution_market_rank import resolve_market_direction
from src.domain.models.trade import TradeDirection


def test_resolve_market_direction_uses_entry_direction():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"raw_prob": 0.80, "execute": True},
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_resolve_market_direction_infers_from_raw_prob():
    entry = {
        "direction": None,
        "metrics": {"raw_prob": 0.20},
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT
