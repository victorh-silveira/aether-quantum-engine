"""Testes do hard-skip (e soft legado) por Edge Cal abaixo do piso."""

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
    assert cfg["neg_edge_hard_skip"] is False
    assert cfg["neg_edge_soft_when_closed_candle_agree"] is True
    assert cfg["neg_edge_soft_min_edge"] == pytest.approx(-1.0)
    assert cfg["neg_edge_bootstrap_soft_kelly_mult"] == pytest.approx(0.25)
    assert cfg["neg_edge_deep_edge_floor"] == pytest.approx(-0.12)
    with pytest.raises(ValueError, match="neg_edge_soft_min_edge"):
        parse_neg_edge_soft_config({"neg_edge_soft_min_edge": 0.05})


def test_neg_edge_hard_blocks_negative_cal_side_edge():
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
        "orchestrator": {
            "execution": {
                "signal_skip": {
                    "neg_edge_soft_kelly_mult": 0.55,
                    "neg_edge_hard_skip": True,
                }
            }
        },
    }
    orch._log_dedupe = {}
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics["signal_status"] == "SKIP:NEG_EDGE"
    assert metrics["gate_reason"] == "neg_edge"
    assert metrics.get("neg_edge_soft") is None
    assert metrics["cal_side_edge"] < 0.0
    assert metrics["kelly_fraction_scale"] == pytest.approx(1.0)
    assert metrics_block_execution(metrics) is True


def test_neg_edge_soft_when_hard_disabled():
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
                    "neg_edge_hard_skip": True,
                }
            }
        },
    }
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics["signal_status"] == "SKIP:NEG_EDGE"
    assert 0.0 < float(metrics["cal_side_edge"]) < 0.04
    assert metrics_block_execution(metrics) is True


def test_neg_edge_nonpositive_hard_blocks_even_when_soft_enabled():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "calibrated_prob": 0.481,
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": True,
        "closed_micro_candle_dir": "CALL",
        "ops_window_candle_dir": "CALL",
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "orchestrator": {
            "execution": {
                "signal_skip": {
                    "neg_edge_soft_kelly_mult": 0.55,
                    "neg_edge_hard_skip": True,
                    "neg_edge_soft_when_closed_candle_agree": False,
                    "neg_edge_soft_min_edge": -1.0,
                    "neg_edge_bootstrap_soft_kelly_mult": 0.25,
                    "neg_edge_deep_edge_floor": -0.12,
                }
            }
        },
    }
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics["gate_reason"] == "neg_edge"
    assert metrics.get("neg_edge_nonpositive_hard") is True
    assert metrics.get("neg_edge_soft") is None
    assert float(metrics["cal_side_edge"]) <= 0.0
    assert metrics_block_execution(metrics) is True


def test_neg_edge_fusion_or_candle_agree_waives_hard_skip():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "calibrated_prob": 0.481,
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": True,
        "closed_micro_candle_dir": "PUT",
        "ops_window_candle_dir": "PUT",
        "neg_edge_fusion_p_eff": 0.88,
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "orchestrator": {
            "execution": {
                "signal_skip": {
                    "neg_edge_soft_kelly_mult": 0.55,
                    "neg_edge_hard_skip": True,
                    "neg_edge_soft_when_closed_candle_agree": False,
                    "neg_edge_soft_min_edge": -1.0,
                    "neg_edge_bootstrap_soft_kelly_mult": 0.25,
                    "neg_edge_deep_edge_floor": -0.12,
                }
            }
        },
    }
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics["signal_status"] == "SKIP:NEG_EDGE"
    assert metrics_block_execution(metrics) is True


def test_neg_edge_hard_skip_positive_subfloor_without_allow_soft():
    metrics = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.59,
        "kelly_fraction_scale": 1.0,
        "loss_clf_auto_learn": True,
        "closed_micro_candle_dir": "PUT",
        "ops_window_candle_dir": "PUT",
    }
    orch = MagicMock()
    orch.config = {
        "deep_learning": {"min_edge_execute": 0.04},
        "risk_management": {"params": {"payout_estimate": 0.72}},
        "orchestrator": {
            "execution": {
                "signal_skip": {
                    "neg_edge_soft_kelly_mult": 0.55,
                    "neg_edge_hard_skip": True,
                    "neg_edge_soft_when_closed_candle_agree": True,
                    "neg_edge_soft_min_edge": -1.0,
                }
            }
        },
    }
    orch._log_dedupe = {}
    assert apply_negative_cal_edge_pause(metrics, orch=orch) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics["gate_reason"] == "neg_edge"
    assert metrics.get("neg_edge_soft") is None
    assert 0.0 < float(metrics["cal_side_edge"]) < 0.04
    assert metrics_block_execution(metrics) is True


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
    assert metrics.get("gate_reason") is None


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


def test_neg_edge_zscore_panic_veto_bilateral():
    """Testa trava de panico: CALL vetado se Z < -2.0 e PUT vetado se Z > +2.0."""
    metrics_call = {
        "execution_candidate_ready": True,
        "exec_direction": "CALL",
        "calibrated_prob": 0.80,
        "edge_zscore": -2.45,
    }
    assert apply_negative_cal_edge_pause(metrics_call, min_edge=0.01, payout=0.95) is True
    assert metrics_call["execution_candidate_ready"] is False
    assert metrics_call["gate_reason"] == "neg_edge_zscore_panic"
    assert metrics_call["signal_status"] == "SKIP:NEG_EDGE_ZSCORE_PANIC"

    metrics_put = {
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
        "calibrated_prob": 0.80,
        "edge_zscore": +2.30,
    }
    assert apply_negative_cal_edge_pause(metrics_put, min_edge=0.01, payout=0.95) is True
    assert metrics_put["execution_candidate_ready"] is False
    assert metrics_put["gate_reason"] == "neg_edge_zscore_panic"
    assert metrics_put["signal_status"] == "SKIP:NEG_EDGE_ZSCORE_PANIC"

