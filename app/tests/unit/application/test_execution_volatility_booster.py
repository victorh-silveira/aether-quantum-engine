from src.application.services.execution_quality_gate import passes_execution_quality
from src.application.services.execution_volatility_booster import (
    apply_volatility_vol_booster,
    volatility_burst_active,
)


def test_volatility_burst_active_requires_macro_and_micro_expansion():
    metrics = {
        "macro_indicators": {"vol_ratio": 1.30},
        "indicators": {"bb_width": 0.025, "vol_ratio": 1.10},
    }
    assert volatility_burst_active(metrics)


def test_volatility_burst_inactive_when_macro_vol_low():
    metrics = {
        "macro_indicators": {"vol_ratio": 1.10},
        "indicators": {"bb_width": 0.03, "vol_ratio": 1.10},
    }
    assert not volatility_burst_active(metrics)


def test_apply_volatility_vol_booster_relaxes_floors_on_burst():
    metrics = {
        "macro_indicators": {"vol_ratio": 1.30},
        "indicators": {"bb_width": 0.03},
    }
    mandatory, edge = apply_volatility_vol_booster(
        metrics,
        mandatory_min_trade_score=0.68,
        min_edge_execute=0.04,
    )
    assert mandatory == 0.65
    assert edge == 0.03
    assert metrics["volatility_vol_booster"] is True


def test_apply_volatility_vol_booster_keeps_lower_existing_floors():
    metrics = {
        "macro_indicators": {"vol_ratio": 1.40},
        "indicators": {"bb_width": 0.04},
    }
    mandatory, edge = apply_volatility_vol_booster(
        metrics,
        mandatory_min_trade_score=0.62,
        min_edge_execute=0.02,
    )
    assert mandatory == 0.62
    assert edge == 0.02


def test_vol_booster_lowers_floors_and_quality_gate_vetoes_weak_edge():
    metrics = {
        "trade_score": 0.68,
        "val_accuracy": 0.70,
        "edge": 0.035,
        "calibrated_prob": 0.70,
        "predicted_payoff_edge": 0.01,
        "macro_indicators": {"vol_ratio": 1.30},
        "indicators": {"bb_width": 0.03, "adx": 0.25},
    }
    boosted_signal, boosted_edge = apply_volatility_vol_booster(
        metrics,
        mandatory_min_trade_score=0.68,
        min_edge_execute=0.04,
    )
    assert boosted_signal < 0.68
    assert boosted_edge < 0.04
    assert passes_execution_quality(metrics, min_direction_margin=0.12, min_payoff_edge=0.04) is True
