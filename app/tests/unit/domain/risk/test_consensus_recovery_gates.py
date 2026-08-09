"""Testes dos gates de EXPLORE no soft recovery."""

from src.domain.risk.consensus_recovery_gates import (
    acc_below_recovery_floor,
    adapted_blocks_dal,
    chop_neg_edge_dampens_dal,
    live_evidence_blocks_dal,
    metric_hurst,
)
from src.domain.risk.soft_recovery_config import load_soft_recovery_from_settings


def test_metric_hurst_reads_indicators_bucket():
    assert metric_hurst({"indicators": {"hurst": 0.41}}) == 0.41
    assert metric_hurst({"regime_chop_hurst": 0.39}) == 0.39
    assert metric_hurst({"hurst": 0.5}) == 0.5
    assert metric_hurst({}) is None
    assert metric_hurst(None) is None
    assert metric_hurst({"hurst": "bad", "indicators": {"hurst": "nope"}}) is None
    assert metric_hurst({"hurst": "bad", "indicators": {"hurst": 0.44}}) == 0.44


def test_recovery_gate_helpers_branches():
    soft = load_soft_recovery_from_settings()
    assert acc_below_recovery_floor({"val_accuracy": 0.01}, 3) is True
    assert live_evidence_blocks_dal({"live_n": 20, "live_wr": 0.40}, 3, soft) is True
    assert adapted_blocks_dal({"scale_adapted": True}, int(soft["adapted_force_explore_linear_min"]), soft) is True


def test_chop_neg_edge_dampens_dal_flags():
    assert chop_neg_edge_dampens_dal({"regime_chop_soft": True, "neg_edge_soft": True}) is True
    assert chop_neg_edge_dampens_dal({"regime_chop_soft": True, "signal_skip_waived": "neg_edge_soft"}) is True
    assert chop_neg_edge_dampens_dal({"neg_edge_soft": True}) is True
    assert chop_neg_edge_dampens_dal({"signal_skip_waived": "neg_edge_soft"}) is True
    assert chop_neg_edge_dampens_dal({"regime_chop_soft": True}) is False
    assert chop_neg_edge_dampens_dal({}) is False
