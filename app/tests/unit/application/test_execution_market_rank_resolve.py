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


def test_resolve_market_direction_trend_exhaustion_put_ignored():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.80,
            "execute": False,
            "trend_direction": "PUT",
            "indicators": {
                "vol_ratio": 1.44,
                "adx": 0.15,
                "rsi": 0.44,
                "keltner": 0.28,
            },
        },
    }
    # A tendência seria PUT, mas como está oversold (rsi < 0.45, keltner < 0.30), ignora e retorna dl_dir (CALL)
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_resolve_market_direction_trend_exhaustion_call_ignored():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.20,
            "execute": False,
            "trend_direction": "CALL",
            "indicators": {
                "vol_ratio": 1.44,
                "adx": 0.15,
                "rsi": 0.58,
                "keltner": 0.75,
            },
        },
    }
    # A tendência seria CALL, mas como está overbought (rsi > 0.55, keltner > 0.70), ignora e retorna dl_dir (PUT)
    assert resolve_market_direction(entry) == TradeDirection.PUT
