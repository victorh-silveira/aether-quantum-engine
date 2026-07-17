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


def _entry(*, direction=TradeDirection.CALL, calibrated_prob=0.72):
    return {
        "direction": direction,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "raw_prob": calibrated_prob,
            "calibrated_prob": calibrated_prob,
            "val_accuracy": 0.70,
            "predicted_payoff_edge": 0.20,
            "meta_classifier_applied": False,
            "meta_payoff_edge_zscore": 0.10,
            "edge_zscore": 0.10,
            "edge_zscore_samples": 12,
        },
    }


def test_resolve_require_meta_for_execution_blocks_without_meta():
    entry = _entry()
    with patch(
        "src.application.services.execution_direction_resolver.resolve_meta_payoff_edge",
        return_value=(0.20, False),
    ):
        result = resolve_execution_direction(
            entry,
            exec_cfg={"require_meta_for_execution": True},
            symbol="RDBULL",
        )
    assert result is not None
    assert entry["metrics"].get("gate_reason") != "meta_unavailable"
    assert entry["metrics"].get("quality_guard_reject") is not True


def test_resolve_recovery_soft_quality_continues_on_borderline_zscore():
    entry = _entry(calibrated_prob=0.51)
    entry["metrics"]["meta_classifier_applied"] = True
    entry["metrics"]["predicted_payoff_edge"] = 0.02
    entry["metrics"]["meta_payoff_edge_zscore"] = -0.10
    entry["metrics"]["edge_zscore"] = -0.10
    result = resolve_execution_direction(
        entry,
        exec_cfg={"quality_gate": {"min_direction_margin": 0.12, "min_meta_payoff_zscore": 0.5}},
        recovery_active=True,
        symbol="RDBULL",
    )
    assert result is not None
    assert result[0] == TradeDirection.CALL
