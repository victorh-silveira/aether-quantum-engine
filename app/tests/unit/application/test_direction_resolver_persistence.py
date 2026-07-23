from types import SimpleNamespace

import pytest

from src.application.services.direction_loss_tracker import (
    record_direction_outcome,
    reset_direction_persistence_tracker,
)
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.payoff_edge_zscore import reset_payoff_edge_buffer
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_edge_buffer():
    reset_payoff_edge_buffer()
    yield
    reset_payoff_edge_buffer()


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


def test_recovery_allows_opposite_side_after_loss_with_positive_edge():
    reset_direction_persistence_tracker()
    record_direction_outcome("R_10", "PUT", won=False)
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = 0.12
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["edge_expectancy"] = "WIN_EXPECTED"
    entry["metrics"]["edge_zscore"] = 0.80
    entry["metrics"]["meta_payoff_edge_zscore"] = 0.80
    entry["metrics"]["edge_zscore_samples"] = 20
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
        recovery_active=True,
        exec_cfg={
            "quality_gate": {
                "min_direction_margin": 0.0,
                "min_payoff_edge": 0.0,
                "regular": {"min_direction_margin": 0.0, "min_payoff_edge": 0.0},
            }
        },
    )
    assert result is not None
    assert result[0] == TradeDirection.CALL
    reset_direction_persistence_tracker()


def test_resolve_keeps_put_when_meta_edge_positive_against_call_zone():
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.35)
    entry["metrics"]["predicted_payoff_edge"] = 0.08
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["bb_pct_b"] = 0.20
    entry["metrics"]["keltner"] = 0.20
    entry["metrics"]["edge_zscore"] = 0.50
    entry["metrics"]["meta_payoff_edge_zscore"] = 0.50
    entry["metrics"]["edge_zscore_samples"] = 20
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
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
    assert result[0] == TradeDirection.PUT
    assert result[1].get("price_zone_kept_meta_side") is True


def test_resolve_skips_repeat_work_same_cycle_after_reject():
    orch = SimpleNamespace(_active_cycle_id=9, _log_dedupe={}, _side_eq_log_keys=set())
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.35)
    entry["metrics"]["predicted_payoff_edge"] = -0.20
    entry["metrics"]["quality_guard_reject"] = True
    entry["metrics"]["gate_reason"] = "meta_negative_edge"
    entry["metrics"]["_direction_resolved_cycle"] = 9
    entry["metrics"]["_recovery_reresolve_done"] = True
    assert resolve_execution_direction(entry, symbol="R_10", orch=orch, cycle_id=9, recovery_active=True) is None
    assert entry["metrics"]["gate_reason"] == "meta_negative_edge"


def test_resolve_reresolves_waivable_gate_once_under_recovery():
    orch = SimpleNamespace(_active_cycle_id=12, _log_dedupe={}, _side_eq_log_keys=set())
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.62)
    entry["metrics"]["predicted_payoff_edge"] = 0.12
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["quality_guard_reject"] = True
    entry["metrics"]["gate_reason"] = "meta_negative_edge"
    entry["metrics"]["_direction_resolved_cycle"] = 12
    entry["metrics"]["edge_zscore"] = 0.40
    entry["metrics"]["meta_payoff_edge_zscore"] = 0.40
    entry["metrics"]["edge_zscore_samples"] = 20
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
        orch=orch,
        cycle_id=12,
        recovery_active=True,
        exec_cfg={
            "quality_gate": {
                "min_direction_margin": 0.0,
                "min_payoff_edge": 0.0,
                "regular": {"min_direction_margin": 0.0, "min_payoff_edge": 0.0},
            }
        },
    )
    assert entry["metrics"].get("_recovery_reresolve_done") is True
    assert result is not None or entry["metrics"].get("gate_reason") != "meta_negative_edge"


def test_persistence_flips_to_opposite_instead_of_deadlock():
    reset_direction_persistence_tracker()
    record_direction_outcome("R_10", "PUT", won=False)
    record_direction_outcome("R_10", "PUT", won=False)
    entry = _entry(direction=TradeDirection.PUT, calibrated_prob=0.35)
    entry["metrics"]["predicted_payoff_edge"] = 0.15
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["edge_zscore"] = 0.80
    entry["metrics"]["meta_payoff_edge_zscore"] = 0.80
    entry["metrics"]["edge_zscore_samples"] = 20
    result = resolve_execution_direction(
        entry,
        symbol="R_10",
        recovery_active=True,
        cycle_id=11,
        exec_cfg={
            "quality_gate": {
                "min_direction_margin": 0.0,
                "min_payoff_edge": 0.0,
                "regular": {"min_direction_margin": 0.0, "min_payoff_edge": 0.0},
            }
        },
    )
    assert result is not None
    assert result[0] == TradeDirection.CALL
    assert result[1].get("persistence_guard_flip") == "CALL"
    assert result[1].get("side_eq_toxic_escape") is True
    reset_direction_persistence_tracker()
