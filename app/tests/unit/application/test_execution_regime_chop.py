"""Testes da atenuacao soft Kelly em regime chop (ADX + Hurst)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.application.services.execution_regime_chop import (
    apply_regime_chop_pause,
    parse_regime_chop_config,
)
from src.application.services.execution_signal_skip import metrics_block_execution, parse_signal_skip_config


def test_parse_regime_chop_from_ssot():
    cfg = parse_regime_chop_config({})
    assert cfg["chop_pause_enabled"] is True
    assert cfg["chop_adx_max"] == pytest.approx(0.22)
    assert cfg["chop_hurst_min"] == pytest.approx(0.47)
    assert cfg["chop_hurst_max"] == pytest.approx(0.53)
    assert cfg["chop_soft_kelly_mult"] == pytest.approx(0.55)
    skip = parse_signal_skip_config({})
    assert skip["chop_pause_enabled"] is True
    assert skip["chop_soft_kelly_mult"] == pytest.approx(0.55)
    assert skip["neg_edge_soft_kelly_mult"] == pytest.approx(0.55)
    assert skip["neg_edge_hard_skip"] is True
    with pytest.raises(ValueError, match="chop_hurst_max"):
        parse_regime_chop_config({"chop_hurst_min": 0.60, "chop_hurst_max": 0.40})
    with pytest.raises(ValueError, match="chop_soft_kelly_mult"):
        parse_regime_chop_config({"chop_soft_kelly_mult": 0.0})


def test_chop_soft_attenuates_random_walk_band():
    metrics = {
        "execution_candidate_ready": True,
        "kelly_fraction_scale": 1.0,
        "indicators": {"adx": 0.20, "hurst": 0.50},
    }
    orch = MagicMock()
    orch._log_dedupe = {}
    assert apply_regime_chop_pause(metrics, orch=orch) is True
    assert metrics["execution_candidate_ready"] is True
    assert metrics.get("signal_status") != "SKIP:REGIME_CHOP"
    assert metrics.get("gate_reason") is None
    assert metrics["regime_chop_soft"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)
    assert metrics_block_execution(metrics) is False


def test_chop_soft_via_scale_micro_when_hurst_outside_band():
    metrics = {
        "execution_candidate_ready": True,
        "kelly_fraction_scale": 1.0,
        "indicators": {"adx": 0.18, "hurst": 0.56},
        "scale_micro_regime": "chop",
    }
    assert apply_regime_chop_pause(metrics) is True
    assert metrics["regime_chop_via_scale"] is True
    assert metrics["regime_chop_soft"] is True
    assert metrics["execution_candidate_ready"] is True


def test_chop_soft_skips_outside_band_or_missing():
    assert (
        apply_regime_chop_pause({"execution_candidate_ready": True, "indicators": {"adx": 0.30, "hurst": 0.50}})
        is False
    )
    assert (
        apply_regime_chop_pause({"execution_candidate_ready": True, "indicators": {"adx": 0.10, "hurst": 0.42}})
        is False
    )
    assert apply_regime_chop_pause({"execution_candidate_ready": True, "indicators": {}}) is False
    micro = {
        "execution_candidate_ready": True,
        "kelly_fraction_scale": 1.0,
        "micro_indicators": {"adx": 0.15, "hurst": 0.49},
    }
    assert apply_regime_chop_pause(micro) is True
    assert micro["regime_chop_soft"] is True


def test_chop_force_and_disabled_and_already_skipped():
    metrics = {
        "execution_candidate_ready": True,
        "indicators": {"adx": 0.10, "hurst": 0.50},
        "kelly_fraction_scale": 1.0,
    }
    assert apply_regime_chop_pause(metrics, force=True) is False
    assert apply_regime_chop_pause(metrics, cfg={"chop_pause_enabled": False}) is False
    blocked = {
        "execution_candidate_ready": False,
        "signal_status": "SKIP:LOSS_CLF_HARD",
        "indicators": {"adx": 0.10, "hurst": 0.50},
    }
    assert apply_regime_chop_pause(blocked) is False
    plain_skip = {
        "execution_candidate_ready": True,
        "signal_status": "SKIP",
        "indicators": {"adx": 0.10, "hurst": 0.50},
    }
    assert apply_regime_chop_pause(plain_skip) is False
    bad_ind = {
        "execution_candidate_ready": True,
        "indicators": {"adx": "x", "hurst": 0.50},
    }
    assert apply_regime_chop_pause(bad_ind) is False
    with_orch = SimpleNamespace(_log_dedupe={})
    assert (
        apply_regime_chop_pause(
            {"execution_candidate_ready": True, "indicators": {"adx": None, "hurst": 0.50}},
            orch=with_orch,
        )
        is False
    )
