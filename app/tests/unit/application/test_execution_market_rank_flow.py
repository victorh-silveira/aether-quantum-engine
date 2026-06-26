from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_market_rank import market_decision_score
from src.domain.models.trade import TradeDirection


def test_market_decision_score_recovery_bonus_for_core_symbol():
    metrics = {
        "raw_prob": 0.62,
        "val_accuracy": 0.60,
        "edge": 0.12,
        "execute": True,
        "deploy_ok": True,
    }
    normal = market_decision_score(metrics, recovery_active=False, symbol="R_50")
    recovery = market_decision_score(
        metrics,
        recovery_active=True,
        symbol="R_50",
        exec_direction=TradeDirection.CALL,
    )
    assert recovery > normal


def test_resolve_execution_direction_strong_call():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "raw_prob": 0.82,
            "trade_score": 0.82,
            "val_accuracy": 0.70,
            "deploy_ok": True,
            "trend_direction": "CALL",
            "indicators": {"hurst": 0.55, "adx": 0.30, "vol_ratio": 1.1, "rsi": 0.52, "keltner": 0.55, "cmo": 0.05},
        },
    }
    result = resolve_execution_direction(entry)
    assert result is not None
    direction, _ = result
    assert direction == TradeDirection.CALL
