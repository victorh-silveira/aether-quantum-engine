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
    assert metrics["execution_candidate_ready"] is False
    assert metrics["signal_status"] == "SKIP:NEG_EDGE"
    assert metrics.get("gate_reason") == "neg_edge"
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
    assert metrics2["gate_reason"] == "neg_edge"
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
    assert metrics3["gate_reason"] == "neg_edge"


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
    assert metrics["gate_reason"] == "neg_edge"
    assert metrics["execution_candidate_ready"] is False


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
    assert metrics.get("neg_edge_fusion_p_eff") is None
    assert isinstance(edge, float)
    assert edge == pytest.approx((0.60 * 1.72) - 1.0)


def test_neg_edge_replay_c1_put_cluster_neg_fusion_pos():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "tcn_direction": "PUT",
        "calibrated_prob": 0.482,
        "fusion_applied": True,
        "fusion_p_eff": 0.612,
        "fusion_reason": "ev_put",
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": False,
        "ops_window_candle_dir": "PUT",
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
    }
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics.get("gate_reason") == "neg_edge"
    assert float(metrics["cal_side_edge"]) == pytest.approx(-0.109, abs=0.002)


def test_neg_edge_replay_c2_call_cluster_neg_fusion_pos():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "tcn_direction": "CALL",
        "calibrated_prob": 0.50509,
        "fusion_applied": True,
        "fusion_p_eff": 0.626,
        "fusion_reason": "ev_call",
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": False,
        "ops_window_candle_dir": "CALL",
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
    }
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics.get("gate_reason") == "neg_edge"
    assert float(metrics["cal_side_edge"]) == pytest.approx(-0.131, abs=0.003)


def test_neg_edge_passes_when_tcn_cal_edge_above_floor():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.70,
        "fusion_applied": True,
        "fusion_p_eff": 0.72,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_negative_cal_edge_pause(metrics, min_edge=0.04, payout=0.72) is False
    assert metrics.get("gate_reason") is None
    assert metrics["execution_candidate_ready"] is True
    assert float(metrics["cal_side_edge"]) >= 0.04
    assert metrics.get("neg_edge_fusion_p_eff") == pytest.approx(0.72)


def test_neg_edge_ignores_fusion_p_eff_out_of_range():
    from src.application.services.execution_neg_edge import _resolve_neg_side_edge

    metrics = {
        "fusion_applied": True,
        "fusion_p_eff": 1.2,
        "calibrated_prob": 0.70,
        "exec_direction": "CALL",
    }
    edge = _resolve_neg_side_edge(metrics, "CALL", 0.72)
    assert metrics.get("neg_edge_fusion_p_eff") is None
    assert edge == pytest.approx((0.70 * 1.72) - 1.0)


def test_neg_edge_hard_without_fusion_blocked_when_p_eff_also_neg():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.50,
        "fusion_applied": True,
        "fusion_p_eff": 0.50,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_negative_cal_edge_pause(metrics, min_edge=0.04, payout=0.72) is True
    assert metrics.get("gate_reason") == "neg_edge"
    assert metrics.get("neg_edge_fusion_blocked") is not True
