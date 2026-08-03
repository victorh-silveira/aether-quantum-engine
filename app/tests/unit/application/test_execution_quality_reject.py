"""Cobertura de waivers de margem e rejeicao cal_margin_floor."""

from __future__ import annotations

from src.application.services.execution_quality_gate_margin import apply_quality_margin_floor_waivers
from src.application.services.execution_quality_reject import has_meta_zscore_telemetry, reject_on_quality_gate


def test_margin_floor_waiver_by_senior_conviction():
    metrics: dict = {"senior_trader_conviction": 0.60}
    assert apply_quality_margin_floor_waivers(metrics, 0.08, exec_cfg=None) == 0.0
    assert metrics.get("quality_margin_senior_waiver") is True


def test_margin_floor_waiver_by_meta_zscore():
    metrics: dict = {"meta_payoff_edge_zscore": 2.5}
    exec_cfg = {"quality_gate": {"min_meta_payoff_zscore": 2.0}}
    assert apply_quality_margin_floor_waivers(metrics, 0.08, exec_cfg=exec_cfg) == 0.0
    assert metrics.get("quality_margin_meta_z_waiver") is True


def test_margin_floor_keeps_floor_without_waiver():
    metrics: dict = {"senior_trader_conviction": 0.40, "edge_zscore": 0.5}
    exec_cfg = {"quality_gate": {"min_meta_payoff_zscore": 2.0}}
    assert apply_quality_margin_floor_waivers(metrics, 0.08, exec_cfg=exec_cfg) == 0.08
    assert "quality_margin_senior_waiver" not in metrics
    assert "quality_margin_meta_z_waiver" not in metrics


def test_reject_on_quality_gate_cal_margin_floor():
    metrics = {"calibrated_prob": 0.52, "raw_prob": 0.52, "val_accuracy": 0.70}
    gate_probe = dict(metrics)
    rejected = reject_on_quality_gate(
        {},
        metrics,
        gate_probe,
        {"hard_cal_margin_floor": 0.04, "quality_gate": {"regular": {"min_direction_margin": 0.0}}},
    )
    assert rejected is True
    assert metrics["gate_reason"] == "cal_margin_floor"


def test_reject_on_quality_gate_skips_floor_in_recovery():
    metrics = {"calibrated_prob": 0.52, "raw_prob": 0.52, "val_accuracy": 0.70}
    gate_probe = dict(metrics)
    rejected = reject_on_quality_gate(
        {},
        metrics,
        gate_probe,
        {"hard_cal_margin_floor": 0.04, "quality_gate": {"regular": {"min_direction_margin": 0.0}}},
        recovery_active=True,
    )
    assert rejected is False
    assert metrics.get("gate_reason") != "cal_margin_floor"


def test_reject_on_quality_gate_skips_floor_with_senior():
    metrics = {
        "calibrated_prob": 0.52,
        "raw_prob": 0.52,
        "val_accuracy": 0.70,
        "senior_trader_conviction": 0.70,
    }
    gate_probe = dict(metrics)
    rejected = reject_on_quality_gate(
        {},
        metrics,
        gate_probe,
        {"hard_cal_margin_floor": 0.04, "quality_gate": {"regular": {"min_direction_margin": 0.0}}},
    )
    assert rejected is False


def test_has_meta_zscore_telemetry_requires_samples():
    assert has_meta_zscore_telemetry({"edge_zscore": 0.2, "edge_zscore_samples": 1}) is False
    assert has_meta_zscore_telemetry({"edge_zscore": 0.2}) is True


def test_reject_on_quality_gate_force_trade_clears_flags():
    metrics = {"quality_guard_reject": True, "regime_skip_cycle": True, "quality_gate_reason": "x"}
    gate_probe = dict(metrics)
    rejected = reject_on_quality_gate(
        {},
        metrics,
        gate_probe,
        {"force_trade_every_cycle": True, "hard_cal_margin_floor": 0.04},
    )
    assert rejected is False
    assert "quality_guard_reject" not in metrics
    assert "quality_gate_reason" not in gate_probe


def test_reject_on_quality_gate_invalid_z_keeps_floor():
    metrics = {
        "calibrated_prob": 0.52,
        "meta_payoff_edge_zscore": object(),
        "edge_zscore_samples": 1,
        "val_accuracy": 0.70,
    }
    gate_probe = dict(metrics)
    rejected = reject_on_quality_gate(
        {},
        metrics,
        gate_probe,
        {
            "hard_cal_margin_floor": 0.04,
            "quality_gate": {
                "min_meta_payoff_zscore": 0.5,
                "regular": {"min_direction_margin": 0.0},
            },
        },
    )
    assert rejected is True
    assert metrics["gate_reason"] == "cal_margin_floor"


def test_reject_on_quality_gate_evaluates_meta_telemetry():
    metrics = {
        "calibrated_prob": 0.70,
        "raw_prob": 0.70,
        "val_accuracy": 0.70,
        "predicted_payoff_edge": 0.10,
        "meta_classifier_applied": True,
        "meta_payoff_edge_zscore": 1.0,
        "edge_zscore": 1.0,
        "edge_zscore_samples": 5,
    }
    gate_probe = dict(metrics)
    rejected = reject_on_quality_gate(
        {},
        metrics,
        gate_probe,
        {"hard_cal_margin_floor": 0.0, "quality_gate": {"regular": {"min_direction_margin": 0.0}}},
    )
    assert rejected is False
