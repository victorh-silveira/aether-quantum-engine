from src.application.services.execution_market_rank import (
    build_market_execution_candidate,
    mandatory_pool_eligible,
    market_decision_score,
    resolve_market_direction,
)
from src.domain.models.trade import TradeDirection


def test_resolve_market_direction_uses_entry_direction():
    entry = {"direction": TradeDirection.CALL, "metrics": {"raw_prob": 0.80}}
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_market_decision_score_uses_raw_and_val_accuracy():
    metrics = {
        "trade_score": 0.80,
        "raw_prob": 0.80,
        "val_accuracy": 0.55,
        "execute": True,
        "deploy_ok": True,
    }
    score = market_decision_score(
        metrics,
        exec_direction=TradeDirection.CALL,
        recovery_active=True,
        symbol="R_50",
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
    )
    assert score > 0.5


def test_mandatory_pool_eligible_rejects_data_gate():
    entry = {"direction": TradeDirection.CALL, "metrics": {"gate_reason": "data"}}
    assert mandatory_pool_eligible(entry) is False
    assert mandatory_pool_eligible({"direction": TradeDirection.PUT, "metrics": {"raw_prob": 0.20}}) is True


def test_build_market_execution_candidate():
    entry = {"direction": TradeDirection.PUT, "metrics": {"raw_prob": 0.20, "execute": True}}
    built = build_market_execution_candidate("R_25", entry)
    assert built is not None
    assert built[1] == TradeDirection.PUT
