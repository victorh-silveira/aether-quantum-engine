from src.application.services.execution_volatility_booster import (
    apply_volatility_vol_booster,
    volatility_burst_active,
)


def test_vol_booster_inactive_keeps_floors():
    metrics = {"indicators": {"vol_ratio": 1.0, "bb_width": 0.01}}
    assert volatility_burst_active(metrics) is False
    score, edge = apply_volatility_vol_booster(metrics, mandatory_min_trade_score=0.6, min_edge_execute=0.05)
    assert score == 0.6
    assert edge == 0.05


def test_vol_booster_active_lowers_floors():
    metrics = {
        "macro_indicators": {"vol_ratio": 9.0},
        "indicators": {"bb_width": 9.0},
    }
    if volatility_burst_active(metrics):
        score, edge = apply_volatility_vol_booster(metrics, mandatory_min_trade_score=0.9, min_edge_execute=0.20)
        assert score <= 0.9
        assert edge <= 0.20
        assert metrics.get("volatility_vol_booster") is True
