from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.payoff_edge_zscore import reset_payoff_edge_buffer
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_edge_buffer():
    reset_payoff_edge_buffer()
    yield
    reset_payoff_edge_buffer()


def _stamp_negative_zscore(metrics: dict, z_score: float = -0.77) -> None:
    metrics["meta_payoff_edge_zscore"] = z_score
    metrics["edge_zscore"] = z_score


def _entry(*, direction=None, raw_prob=0.55, calibrated_prob=None, execute=True, gate_reason=None, deploy_ok=True):
    metrics = {
        "execute": execute,
        "gate_reason": gate_reason,
        "deploy_ok": deploy_ok,
        "raw_prob": raw_prob,
        "val_accuracy": 0.70,
        "predicted_payoff_edge": 0.06,
        "meta_classifier_applied": True,
    }
    if calibrated_prob is not None:
        metrics["calibrated_prob"] = calibrated_prob
    return {"direction": direction, "metrics": metrics}


def test_resolve_mild_negative_edge_is_blocked():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = -0.08
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["edge_expectancy"] = "LOSS_EXPECTED"
    entry["metrics"]["indicators"] = {"bb_width": 0.09}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.04}
    with patch(
        "src.application.services.execution_direction_resolver.attach_payoff_edge_zscore_metrics",
        side_effect=lambda metrics, edge, **kwargs: _stamp_negative_zscore(metrics),
    ):
        result = resolve_execution_direction(
            entry,
            symbol="R_10",
            exec_cfg={"quality_gate": {"min_payoff_edge": 0.0, "regular": {"min_payoff_edge": 0.0}}},
        )
    assert result is None


def test_resolve_mild_negative_edge_blocked_even_with_mandatory_flag():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.46)
    entry["metrics"]["predicted_payoff_edge"] = -0.15
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.09, "adx": 0.25, "rsi": 0.4}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.04}
    entry["metrics"]["edge_zscore"] = 0.0
    entry["metrics"]["meta_payoff_edge_zscore"] = 0.0
    entry["metrics"]["edge_zscore_samples"] = 20
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
        skipped_cycles_counter=0,
        exec_cfg={
            "mandatory_trade_each_cycle": True,
            "quality_gate": {
                "min_direction_margin": 0.03,
                "min_payoff_edge": 0.02,
                "regular": {"min_direction_margin": 0.03, "min_payoff_edge": 0.02},
            },
        },
    )
    assert result is None


def test_resolve_thin_put_margin_rejected_by_quality_floor():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.490)
    entry["metrics"]["predicted_payoff_edge"] = 0.06
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.09, "adx": 0.25, "rsi": 0.4}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.04}
    entry["metrics"]["edge_zscore"] = 0.0
    entry["metrics"]["meta_payoff_edge_zscore"] = 0.0
    entry["metrics"]["edge_zscore_samples"] = 20
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
        skipped_cycles_counter=0,
        exec_cfg={
            "quality_gate": {
                "min_direction_margin": 0.03,
                "min_payoff_edge": 0.02,
                "regular": {"min_direction_margin": 0.03, "min_payoff_edge": 0.02},
            },
        },
    )
    assert result is not None
    assert result[0] == TradeDirection.PUT


def test_resolve_defined_direction_with_positive_edge_passes():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.44)
    entry["metrics"]["predicted_payoff_edge"] = 0.05
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.09, "adx": 0.25, "rsi": 0.4}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.04}
    entry["metrics"]["edge_zscore"] = 0.2
    entry["metrics"]["meta_payoff_edge_zscore"] = 0.2
    entry["metrics"]["edge_zscore_samples"] = 20
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
        skipped_cycles_counter=0,
        exec_cfg={
            "require_meta_for_execution": True,
            "quality_gate": {
                "min_direction_margin": 0.03,
                "min_payoff_edge": 0.02,
                "regular": {"min_direction_margin": 0.03, "min_payoff_edge": 0.02},
            },
        },
    )
    assert result is not None
    assert result[0] == TradeDirection.PUT
    assert entry["metrics"].get("quality_gate_reason") != "direction_margin_gate"
    assert entry["metrics"].get("gate_reason") != "meta_negative_edge"


def test_resolve_negative_edge_allowed_under_starvation_floor():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.46)
    entry["metrics"]["predicted_payoff_edge"] = -0.04
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["edge_expectancy"] = "LOSS_EXPECTED"
    entry["metrics"]["indicators"] = {"bb_width": 0.09, "adx": 0.25, "rsi": 0.4}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.04}
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
        skipped_cycles_counter=16,
        exec_cfg={
            "quality_gate": {
                "min_direction_margin": 0.0,
                "min_payoff_edge": 0.0,
                "regular": {"min_direction_margin": 0.0, "min_payoff_edge": 0.0},
            }
        },
    )
    assert result is not None
    assert result[0] == TradeDirection.PUT


def test_resolve_negative_edge_blocked_in_recovery_with_non_negative_floor():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.46)
    entry["metrics"]["predicted_payoff_edge"] = -0.20
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["edge_expectancy"] = "LOSS_EXPECTED"
    entry["metrics"]["indicators"] = {"bb_width": 0.09, "adx": 0.25, "rsi": 0.4}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.04}
    risk = SimpleNamespace(consecutive_losses_linear=1, pending_loss={"R_10": 16.0}, dlambert_unit=16.0)
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
        recovery_active=True,
        risk_manager=risk,
        skipped_cycles_counter=5,
        exec_cfg={
            "quality_gate": {
                "min_direction_margin": 0.0,
                "min_payoff_edge": 0.02,
                "regular": {"min_direction_margin": 0.0, "min_payoff_edge": 0.02},
                "recovery_relax": {
                    "edge_floor": 0.0,
                    "edge_zscore_waiver": 0.5,
                    "full_pending_units": 8.0,
                    "margin_floor": 0.02,
                    "min_linear": 2,
                    "session_stake_unit_bankroll_pct": 0.0015,
                },
            }
        },
    )
    assert result is None


def test_resolve_aligns_negative_edge_put_to_buy_zone_under_starvation():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.46)
    entry["metrics"]["predicted_payoff_edge"] = -0.04
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["bb_pct_b"] = 0.20
    entry["metrics"]["keltner"] = 0.20
    entry["metrics"]["edge_zscore"] = 0.0
    entry["metrics"]["meta_payoff_edge_zscore"] = 0.0
    entry["metrics"]["edge_zscore_samples"] = 20
    entry["metrics"]["indicators"] = {"bb_width": 0.09, "adx": 0.25, "rsi": 0.4}
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
        skipped_cycles_counter=16,
        exec_cfg={
            "price_zone": {
                "enabled": True,
                "buy_max": 0.48,
                "sell_min": 0.52,
                "bb_weight": 0.6,
                "keltner_weight": 0.4,
                "neutral_mode": "nearest",
                "require_trend_agreement": False,
                "require_tcn_agreement": False,
            },
            "quality_gate": {
                "min_direction_margin": 0.0,
                "min_payoff_edge": 0.0,
                "regular": {"min_direction_margin": 0.0, "min_payoff_edge": 0.0},
            },
        },
    )
    assert result is not None
    assert result[0] == TradeDirection.CALL
    assert result[1].get("price_zone") == "BUY"


def test_recovery_blocks_same_side_after_loss_with_negative_edge():
    from src.application.services.direction_loss_tracker import (
        record_direction_outcome,
        reset_direction_persistence_tracker,
    )

    reset_direction_persistence_tracker()
    record_direction_outcome("R_10", "PUT", won=False)
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.35)
    entry["metrics"]["predicted_payoff_edge"] = -0.12
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["edge_expectancy"] = "LOSS_EXPECTED"
    entry["metrics"]["indicators"] = {"bb_width": 0.09, "adx": 0.25, "rsi": 0.4}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": 0.04}
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
        recovery_active=True,
        exec_cfg={
            "quality_gate": {
                "min_direction_margin": 0.0,
                "min_payoff_edge": 0.02,
                "regular": {"min_direction_margin": 0.0, "min_payoff_edge": 0.02},
                "recovery_relax": {
                    "edge_floor": 0.0,
                    "edge_zscore_waiver": 0.5,
                    "full_pending_units": 8.0,
                    "margin_floor": 0.02,
                    "min_linear": 2,
                    "session_stake_unit_bankroll_pct": 0.0015,
                },
            }
        },
    )
    assert result is None
    reset_direction_persistence_tracker()
