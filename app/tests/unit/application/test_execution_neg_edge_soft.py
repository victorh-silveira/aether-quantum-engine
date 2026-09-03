"""Soft neg_edge: candle-agree, piso de profundidade e p_ovr."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.application.services.execution_neg_edge import apply_negative_cal_edge_pause
from src.application.services.execution_signal_skip import metrics_block_execution, parse_signal_skip_config


def test_parse_signal_skip_rejects_soft_min_edge():
    with pytest.raises(ValueError, match="neg_edge_soft_min_edge"):
        parse_signal_skip_config({"neg_edge_soft_min_edge": 0.1})


def _orch_skip(**overrides):
    block = {
        "neg_edge_soft_kelly_mult": 0.55,
        "neg_edge_hard_skip": True,
        "neg_edge_soft_when_closed_candle_agree": True,
        "neg_edge_soft_min_edge": -0.05,
    }
    block.update(overrides)
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "orchestrator": {"execution": {"signal_skip": block}},
    }
    orch._log_dedupe = {}
    return orch


def test_neg_edge_soft_when_side_agrees_closed_candle():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.59,
        "kelly_fraction_scale": 1.0,
        "closed_micro_candle_dir": "CALL",
        "ops_window_candle_dir": "CALL",
        "loss_clf_auto_learn": True,
    }
    assert apply_negative_cal_edge_pause(metrics, orch=_orch_skip()) is True
    assert metrics.get("gate_reason") == "neg_edge"
    assert 0.0 < float(metrics["cal_side_edge"]) < 0.04
    assert metrics_block_execution(metrics) is True


def test_neg_edge_hard_when_candle_agree_but_edge_nonpositive():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.53,
        "kelly_fraction_scale": 1.0,
        "closed_micro_candle_dir": "CALL",
        "ops_window_candle_dir": "CALL",
    }
    orch = _orch_skip(neg_edge_soft_min_edge=-1.0)
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics.get("gate_reason") == "neg_edge"
    assert metrics["execution_candidate_ready"] is False
    assert metrics_block_execution(metrics) is True


def test_neg_edge_soft_subfloor_when_soft_min_wide():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.59,
        "kelly_fraction_scale": 1.0,
        "closed_micro_candle_dir": "PUT",
        "ops_window_candle_dir": "PUT",
        "loss_clf_auto_learn": True,
    }
    orch = _orch_skip(neg_edge_hard_skip=False, neg_edge_soft_min_edge=-1.0)
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is False
    assert metrics.get("gate_reason") is None
    assert 0.0 < float(metrics["cal_side_edge"]) < 0.04
    assert metrics["execution_candidate_ready"] is True
    assert metrics.get("neg_edge_soft") is True
    assert metrics_block_execution(metrics) is False


def test_neg_edge_soft_when_flip_blocked_and_candle_with_hard_on():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.59,
        "kelly_fraction_scale": 1.0,
        "closed_micro_candle_dir": "CALL",
        "ops_window_candle_dir": "CALL",
        "loss_clf_flip_blocked": "neg_edge",
        "loss_clf_auto_learn": True,
    }
    assert apply_negative_cal_edge_pause(metrics, orch=_orch_skip()) is True
    assert metrics.get("gate_reason") == "neg_edge"
    assert metrics_block_execution(metrics) is True


def test_neg_edge_soft_exec_when_hard_skip_false_and_edge_nonpositive():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.50,
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": True,
    }
    orch = _orch_skip(neg_edge_hard_skip=False, neg_edge_soft_min_edge=-1.0)
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is False
    assert float(metrics["cal_side_edge"]) <= 0.0
    assert metrics["execution_candidate_ready"] is True
    assert metrics.get("neg_edge_soft") is True
    assert metrics.get("signal_skip_waived") == "neg_edge_soft"
    assert metrics.get("gate_reason") is None
    assert metrics_block_execution(metrics) is False


def test_neg_edge_soft_when_loss_clf_p_ovr_flip():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.59,
        "kelly_fraction_scale": 1.0,
        "closed_micro_candle_dir": "PUT",
        "ops_window_candle_dir": "PUT",
        "loss_clf_flip": True,
        "loss_clf_flip_scale_p_override": True,
        "loss_clf_auto_learn": True,
    }
    assert apply_negative_cal_edge_pause(metrics, orch=_orch_skip()) is True
    assert metrics.get("gate_reason") == "neg_edge"
    assert metrics_block_execution(metrics) is True
