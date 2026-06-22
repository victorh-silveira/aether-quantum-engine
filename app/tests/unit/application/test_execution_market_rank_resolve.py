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


def test_resolve_market_direction_mean_reversion_inversion_put_to_call():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.35,
            "indicators": {
                "hurst": 0.45,
                "adx": 0.22,
                "vol_ratio": 0.80,
                "rsi": 0.38,
            },
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL
    assert entry["metrics"]["direction_inverted"] is True


def test_resolve_market_direction_mean_reversion_inversion_call_to_put():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.65,
            "indicators": {
                "hurst": 0.45,
                "adx": 0.22,
                "vol_ratio": 0.80,
                "rsi": 0.62,
            },
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT
    assert entry["metrics"]["direction_inverted"] is True
