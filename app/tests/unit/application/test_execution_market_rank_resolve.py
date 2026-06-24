from src.application.services.execution_direction import _entry_gate_blocked
from src.application.services.execution_market_rank import mandatory_pool_eligible, resolve_market_direction
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


def test_resolve_market_direction_unreliable_accuracy_no_inversion():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.49,
            "raw_prob": 0.80,
            "execute": True,
        },
    }
    # Acurácia de 0.49 está abaixo de 0.50, portanto DEVE inverter (inverte CALL para PUT)
    assert resolve_market_direction(entry) == TradeDirection.PUT
    assert entry["metrics"].get("direction_inverted") is True


def test_mandatory_pool_eligible_grey_zone():
    entry_grey = {
        "direction": TradeDirection.CALL,
        "metrics": {"val_accuracy": 0.49, "raw_prob": 0.80, "execute": False},
    }
    entry_good = {
        "direction": TradeDirection.CALL,
        "metrics": {"val_accuracy": 0.55, "raw_prob": 0.80, "execute": True},
    }
    entry_inverted = {
        "direction": TradeDirection.CALL,
        "metrics": {"val_accuracy": 0.42, "raw_prob": 0.80, "execute": False},
    }
    assert mandatory_pool_eligible(entry_grey) is True
    assert mandatory_pool_eligible(entry_good) is True
    assert mandatory_pool_eligible(entry_inverted) is True


def test_entry_gate_blocked_grey_zone():
    metrics_grey = {"val_accuracy": 0.49, "raw_prob": 0.80, "execute": False}
    metrics_good = {"val_accuracy": 0.55, "raw_prob": 0.80, "execute": True}
    metrics_inverted = {"val_accuracy": 0.42, "raw_prob": 0.80, "execute": False}
    assert _entry_gate_blocked(metrics_grey) is False
    assert _entry_gate_blocked(metrics_good) is False
    assert _entry_gate_blocked(metrics_inverted) is False


def test_resolve_market_direction_mean_reversion_disabled():
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
    # With mean reversion disabled, it should not invert PUT to CALL
    assert resolve_market_direction(entry, mean_reversion_enabled=False) == TradeDirection.PUT
    assert entry["metrics"].get("direction_inverted") is not True


def test_resolve_market_direction_low_accuracy_disabled():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.42,
            "raw_prob": 0.80,
            "trend_direction": "PUT",
            "indicators": {
                "rsi": 0.50,
                "keltner": 0.50,
            },
        },
    }
    # With low accuracy inversion disabled, it should not invert CALL to PUT
    assert resolve_market_direction(entry, low_accuracy_enabled=False) == TradeDirection.CALL
    assert entry["metrics"].get("direction_inverted") is not True
