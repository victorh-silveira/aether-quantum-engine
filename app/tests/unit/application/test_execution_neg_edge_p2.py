"""Testes do hard-skip (e soft legado) por Edge Cal abaixo do piso (parte 2)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.application.services.execution_neg_edge import (
    apply_negative_cal_edge_pause,
)


def test_neg_edge_payout_and_min_edge_fallbacks():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.59,
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": True,
    }
    assert apply_negative_cal_edge_pause(metrics, orch=None, min_edge=0.04, payout=0.72, soft_mult=0.55) is True
    assert metrics["execution_candidate_ready"] is True
    assert metrics["neg_edge_soft"] is True
    assert metrics.get("gate_reason") != "neg_edge"
    bad_orch = MagicMock()
    bad_orch.config = {
        "risk_management": {"params": {"payout_estimate": "x"}},
        "deep_learning": {"min_edge_execute": "y"},
        "orchestrator": {
            "execution": {
                "signal_skip": {
                    "neg_edge_soft_kelly_mult": 0.55,
                    "neg_edge_hard_skip": True,
                }
            }
        },
    }
    metrics2 = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "calibrated_prob": 0.55,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_negative_cal_edge_pause(metrics2, orch=bad_orch) is True
    assert metrics2["gate_reason"] == "neg_edge"


def test_neg_edge_skips_missing_direction_and_skip_status():
    metrics = {"execution_candidate_ready": True, "calibrated_prob": 0.40, "kelly_fraction_scale": 1.0}
    assert apply_negative_cal_edge_pause(metrics, min_edge=0.04, payout=0.72) is False
    skip = {
        "execution_candidate_ready": True,
        "signal_status": "SKIP:TECH",
        "exec_direction": "CALL",
        "calibrated_prob": 0.40,
    }
    assert apply_negative_cal_edge_pause(skip, min_edge=0.04, payout=0.72) is False


def test_neg_edge_hard_clears_prior_soft_waive_and_malformed_orch():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.40,
        "kelly_fraction_scale": 1.0,
        "signal_skip_waived": "neg_edge_soft",
        "neg_edge_soft": True,
    }
    hard_orch = MagicMock()
    hard_orch.config = {
        "orchestrator": {
            "execution": {
                "signal_skip": {
                    "neg_edge_soft_kelly_mult": 0.55,
                    "neg_edge_hard_skip": True,
                    "neg_edge_soft_when_closed_candle_agree": False,
                    "neg_edge_soft_min_edge": -1.0,
                }
            }
        },
    }
    hard_orch._log_dedupe = {}
    assert apply_negative_cal_edge_pause(metrics, orch=hard_orch, min_edge=0.04, payout=0.72) is True
    assert metrics["gate_reason"] == "neg_edge"
    assert metrics.get("signal_skip_waived") is None
    orch2 = MagicMock()
    orch2.config = {"orchestrator": "x"}
    metrics2 = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.59,
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": True,
    }
    assert apply_negative_cal_edge_pause(metrics2, orch=orch2, min_edge=0.04, payout=0.72) is True
    assert metrics2["neg_edge_soft"] is True
    orch3 = MagicMock()
    orch3.config = {"orchestrator": {"execution": "x"}}
    metrics3 = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.59,
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": True,
    }
    assert apply_negative_cal_edge_pause(metrics3, orch=orch3, min_edge=0.04, payout=0.72) is True
    assert metrics3["neg_edge_soft"] is True


def test_neg_edge_soft_mult_override_with_orch():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.59,
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": True,
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "orchestrator": {
            "execution": {
                "signal_skip": {
                    "neg_edge_soft_kelly_mult": 0.55,
                    "neg_edge_hard_skip": False,
                }
            }
        },
    }
    assert apply_negative_cal_edge_pause(metrics, orch=orch, soft_mult=0.40) is True
    assert metrics["neg_edge_soft"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.40)


def test_neg_edge_fusion_p_eff_invalid_falls_back_to_cal():
    from src.application.services.execution_neg_edge import _resolve_neg_side_edge

    metrics = {
        "fusion_applied": True,
        "fusion_p_eff": "bad",
        "calibrated_prob": 0.40,
        "exec_direction": "PUT",
    }
    edge = _resolve_neg_side_edge(metrics, "PUT", 0.72)
    assert metrics.get("neg_edge_used_fusion_p_eff") is not True
    assert isinstance(edge, float)
