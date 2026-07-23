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


def _c0015_entry():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = -0.22
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["indicators"] = {"bb_width": 0.03}
    entry["metrics"]["flow_features"] = {"micro_tick_acceleration": -0.02}
    return entry


def test_resolve_rejects_weak_edge_without_meta_prefetch():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.70)
    entry["metrics"]["predicted_payoff_edge"] = 0.01
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=2,
        pending_loss={},
        pending_loss_total=lambda: 0.0,
    )
    result = resolve_execution_direction(
        entry,
        infra_cfg={"meta_classifier": {"enabled": True}},
        symbol="R_10",
        risk_manager=risk_manager,
    )
    assert result is not None
    assert result[0] == TradeDirection.CALL


def test_resolve_c0015_negative_edge_blocked_by_meta_payoff_veto(caplog):
    entry = _c0015_entry()
    entry["metrics"]["edge_expectancy"] = "LOSS_EXPECTED"
    with (
        patch(
            "src.application.services.execution_direction_resolver.attach_payoff_edge_zscore_metrics",
            side_effect=lambda metrics, edge, **kwargs: _stamp_negative_zscore(metrics),
        ),
        caplog.at_level("INFO"),
    ):
        result = resolve_execution_direction(entry, symbol="R_10")
    assert result is None
    assert entry["metrics"].get("gate_reason") == "meta_negative_edge"
    assert not any("[D-SQUEEZE]" in record.message for record in caplog.records)


def test_resolve_persistence_guard_freeze_skips_without_inverting():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.72)
    with (
        patch(
            "src.application.services.execution_direction_persistence.evaluate_direction_persistence_guard",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_persistence.consecutive_direction_losses",
            return_value=2,
        ),
    ):
        assert (
            resolve_execution_direction(
                entry,
                symbol="R_10",
                cycle_id=11,
                exec_cfg={"direction_persistence": {"same_direction_count_threshold": 2}},
            )
            is None
        )
    assert entry["metrics"].get("persistence_guard_skip") is True
    assert entry["metrics"].get("gate_reason") == "persistence_guard_skip"


def test_resolve_sets_meta_veto_mode_none_when_absent():
    entry = _entry(direction=TradeDirection.CALL, calibrated_prob=0.72)
    entry["metrics"]["predicted_payoff_edge"] = 0.25
    entry["metrics"]["meta_payoff_edge_zscore"] = 1.2
    entry["metrics"]["edge_zscore"] = 1.2
    entry["metrics"]["edge_zscore_samples"] = 20
    with patch(
        "src.application.services.execution_direction_resolver.should_veto_meta_payoff_negative_zscore",
        return_value=False,
    ):
        result = resolve_execution_direction(entry, symbol="R_10")
    assert result is not None
    assert result[1].get("meta_veto_mode") == "none"
