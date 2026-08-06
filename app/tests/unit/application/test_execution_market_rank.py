import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_market_rank import (
    _trade_score,
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
        },
    }
    base.update(metrics)
    return {"direction": direction, "metrics": base}


def test_trade_score_reads_metrics():
    assert _trade_score({"trade_score": 0.7}) == pytest.approx(0.7)


def test_mandatory_pool_eligible_true():
    assert mandatory_pool_eligible(_entry(), exec_cfg={}) is True


def test_mandatory_pool_eligible_blocks_deploy():
    assert mandatory_pool_eligible(_entry(deploy_ok=False), exec_cfg={}) is False


def test_market_decision_score_positive():
    score = market_decision_score(_entry()["metrics"], symbol="OTC_SPC")
    assert score > 0.0


def test_build_market_execution_candidate():
    built = build_market_execution_candidate("OTC_SPC", _entry())
    assert built is not None
    assert built[0] == "OTC_SPC"


def test_resolve_execution_direction_for_rank_entry():
    result = resolve_execution_direction(_entry(), exec_cfg={}, symbol="OTC_SPC")
    assert result is not None
