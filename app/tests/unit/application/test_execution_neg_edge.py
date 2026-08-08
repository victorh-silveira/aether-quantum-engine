"""Testes da atenuacao soft Kelly por Edge Cal abaixo do piso."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.application.services.execution_neg_edge import (
    apply_negative_cal_edge_pause,
    parse_neg_edge_soft_config,
)
from src.application.services.execution_signal_skip import metrics_block_execution


def test_parse_neg_edge_soft_from_ssot():
    cfg = parse_neg_edge_soft_config({})
    assert cfg["neg_edge_soft_kelly_mult"] == pytest.approx(0.55)


def test_neg_edge_soft_attenuates_negative_cal_side_edge():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "calibrated_prob": 0.481,
        "kelly_fraction_scale": 1.0,
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "orchestrator": {"execution": {"signal_skip": {"neg_edge_soft_kelly_mult": 0.55}}},
    }
    orch._log_dedupe = {}
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics["execution_candidate_ready"] is True
    assert metrics.get("signal_status") != "SKIP:NEG_EDGE"
    assert metrics.get("gate_reason") is None
    assert metrics["neg_edge_soft"] is True
    assert metrics["cal_side_edge"] < 0.0
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)
    assert metrics_block_execution(metrics) is False


def test_neg_edge_allows_positive_edge_above_floor():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.70,
        "kelly_fraction_scale": 1.0,
    }
    assert (
        apply_negative_cal_edge_pause(
            metrics,
            min_edge=0.04,
            payout=0.72,
            soft_mult=0.55,
        )
        is False
    )
    assert metrics["cal_side_edge"] >= 0.04
    assert metrics.get("neg_edge_soft") is None


def test_neg_edge_respects_force_and_prior_skip():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "calibrated_prob": 0.40,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_negative_cal_edge_pause(metrics, force=True, min_edge=0.04, payout=0.72) is False
    blocked = {
        "execution_candidate_ready": False,
        "signal_status": "SKIP:REGIME_CHOP",
        "exec_direction": "PUT",
        "calibrated_prob": 0.40,
    }
    assert apply_negative_cal_edge_pause(blocked, min_edge=0.04, payout=0.72) is False


def test_parse_neg_edge_soft_rejects_out_of_range():
    with pytest.raises(ValueError, match="neg_edge_soft_kelly_mult"):
        parse_neg_edge_soft_config({"neg_edge_soft_kelly_mult": 1.5})


def test_neg_edge_payout_and_min_edge_fallbacks():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.40,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_negative_cal_edge_pause(metrics, orch=None, min_edge=0.04, payout=0.72, soft_mult=0.55) is True
    bad_orch = MagicMock()
    bad_orch.config = {
        "risk_management": {"params": {"payout_estimate": "x"}},
        "deep_learning": {"min_edge_execute": "y"},
    }
    metrics2 = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "calibrated_prob": 0.55,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_negative_cal_edge_pause(metrics2, orch=bad_orch) is True


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
