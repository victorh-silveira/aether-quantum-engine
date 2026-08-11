"""Testes do catalogo minimo de atenuacao de sinal (parte 2)."""

from unittest.mock import MagicMock, patch

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.domain.models.trade import TradeDirection


def test_resolve_with_orch_applies_cal_margin_soft():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "raw_prob": 0.51,
            "calibrated_prob": 0.51,
            "val_accuracy": 0.70,
            "predicted_payoff_edge": 0.06,
            "meta_classifier_applied": True,
            "scale_mini_prev_bar_dir": "CALL",
            "scale_mini_bar_dir": "CALL",
            "kelly_fraction_scale": 1.0,
        },
    }
    orch = MagicMock()
    orch.config = {
        "infra": {"loss_classifier": {"enabled": False}},
        "deep_learning": {"min_edge_execute": -0.99},
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "orchestrator": {"execution": {"scale_vision": {"fusion_enabled": False}}},
    }
    orch._active_cycle_id = 0
    orch.risk_manager.pending_loss_total.return_value = 0.0
    with (
        patch(
            "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
            return_value=False,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_negative_cal_edge_pause",
            return_value=False,
        ),
    ):
        result = resolve_execution_direction(entry, symbol="R_10", orch=orch)
    assert result is not None
    _dir, metrics = result
    assert metrics.get("gate_reason") is None
    assert metrics.get("signal_skip_waived") == "cal_margin_soft"
    assert metrics.get("execution_candidate_ready") is not False
