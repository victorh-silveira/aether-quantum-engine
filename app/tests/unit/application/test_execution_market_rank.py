from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_market_rank import (
    build_market_execution_candidate,
    mandatory_pool_eligible,
    market_decision_score,
)
from src.domain.models.trade import TradeDirection


def _entry(direction=TradeDirection.CALL, raw_prob=0.62, **metrics):
    base = {
        "execute": True,
        "deploy_ok": True,
        "raw_prob": raw_prob,
        "trade_score": max(raw_prob, 1.0 - raw_prob),
        "val_accuracy": 0.60,
        "edge": abs(raw_prob - 0.5),
        "trend_direction": "CALL",
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
    return {"direction": direction, "metrics": base}


def test_mandatory_pool_eligible_requires_inferable_direction():
    assert mandatory_pool_eligible(_entry()) is True
    assert mandatory_pool_eligible({"direction": None, "metrics": {"deploy_ok": True}}) is False


def test_market_decision_score_prefers_higher_raw_side():
    low = market_decision_score(_entry(raw_prob=0.52)["metrics"])
    high = market_decision_score(_entry(raw_prob=0.80)["metrics"])
    assert high > low


def test_build_market_execution_candidate_uses_resolver():
    built = build_market_execution_candidate("R_50", _entry(direction=TradeDirection.PUT, raw_prob=0.40))
    assert built is not None
    symbol, direction, metrics = built
    assert symbol == "R_50"
    assert direction in (TradeDirection.CALL, TradeDirection.PUT)
    assert "exec_direction" in metrics


def test_market_decision_score_recovery_indicator_adjustments():
    metrics = {
        "raw_prob": 0.62,
        "val_accuracy": 0.60,
        "edge": 0.12,
        "execute": True,
        "deploy_ok": True,
        "indicators": {"adx": 0.15, "vol_ratio": 0.90, "hurst": 0.40},
    }
    low_adx = market_decision_score(metrics, recovery_active=True, symbol="R_50")
    high_adx = market_decision_score(
        {
            **metrics,
            "indicators": {"adx": 0.30, "vol_ratio": 1.10, "hurst": 0.60},
        },
        recovery_active=True,
        symbol="R_50",
    )
    assert high_adx > low_adx


def test_market_decision_score_penalizes_inverted_and_low_margin():
    aligned = market_decision_score(_entry(raw_prob=0.80, direction_margin=0.10)["metrics"])
    inverted = market_decision_score(_entry(raw_prob=0.80, direction_margin=0.10, direction_inverted=True)["metrics"])
    low_margin = market_decision_score(_entry(raw_prob=0.80, direction_margin=0.03)["metrics"])
    assert aligned > inverted
    assert aligned > low_margin


def test_resolve_flips_on_low_val_accuracy():
    entry = _entry(direction=TradeDirection.CALL, raw_prob=0.58, val_accuracy=0.45, trend_direction="PUT")
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, metrics = result
    assert direction == TradeDirection.PUT
    assert metrics["direction_inverted"] is True
