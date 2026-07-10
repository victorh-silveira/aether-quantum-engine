from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.domain.models.trade import TradeDirection


def _entry(**metrics):
    base = {
        "raw_prob": 0.58,
        "trade_score": 0.58,
        "val_accuracy": 0.60,
        "deploy_ok": True,
        "trend_direction": "CALL",
        "call_votes": 4,
        "put_votes": 2,
        "indicators": {
            "hurst": 0.55,
            "adx": 0.30,
            "vol_ratio": 1.10,
            "rsi": 0.52,
            "keltner": 0.55,
            "cmo": 0.05,
        },
    }
    base.update(metrics)
    return {"direction": TradeDirection.CALL, "metrics": base}


def test_resolve_uses_entry_direction():
    entry = _entry()
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    assert result[0] == TradeDirection.CALL


def test_resolve_infers_from_raw_prob():
    entry = {"direction": None, "metrics": _entry(raw_prob=0.42, trend_direction="PUT")["metrics"]}
    result = resolve_execution_direction(entry, symbol="RDBEAR")
    assert result is not None
    assert result[0] == TradeDirection.PUT


def test_resolve_put_on_bear_with_low_prob():
    entry = _entry(
        raw_prob=0.42,
        trend_direction="PUT",
        predicted_payoff_edge=0.06,
        meta_classifier_applied=True,
        indicators={
            "hurst": 0.50,
            "adx": 0.30,
            "vol_ratio": 1.10,
            "rsi": 0.30,
            "keltner": 0.20,
            "cmo": -0.10,
        },
    )
    entry["direction"] = TradeDirection.PUT
    result = resolve_execution_direction(entry, symbol="RDBEAR")
    assert result is not None
    assert result[0] == TradeDirection.PUT


def test_resolve_low_accuracy_keeps_dl_side():
    entry = _entry(
        direction=TradeDirection.CALL,
        raw_prob=0.58,
        val_accuracy=0.45,
        trend_direction="PUT",
        indicators={
            "hurst": 0.52,
            "adx": 0.18,
            "vol_ratio": 0.90,
            "rsi": 0.50,
            "keltner": 0.55,
            "cmo": 0.05,
        },
    )
    result = resolve_execution_direction(entry, symbol="RDBULL")
    assert result is not None
    assert result[0] == TradeDirection.CALL
    assert result[1]["direction_inverted"] is False
