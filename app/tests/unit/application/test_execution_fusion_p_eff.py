"""Alinhamento fusion_p_eff ao lado EXEC final."""

from __future__ import annotations

import pytest

from src.application.services.execution_fusion_p_eff import (
    resolve_neg_side_edge,
    stamp_fusion_p_eff,
    sync_fusion_p_eff_for_direction,
)


def test_sync_fusion_p_eff_for_direction_call():
    metrics = {"fusion_applied": True, "fusion_p_call": 0.61, "fusion_p_put": 0.39, "fusion_p_eff": 0.39}
    sync_fusion_p_eff_for_direction(metrics, "CALL")
    assert metrics["fusion_p_eff"] == pytest.approx(0.61)


def test_sync_fusion_p_eff_for_direction_put():
    metrics = {"fusion_applied": True, "fusion_p_call": 0.61, "fusion_p_put": 0.39, "fusion_p_eff": 0.61}
    sync_fusion_p_eff_for_direction(metrics, "PUT")
    assert metrics["fusion_p_eff"] == pytest.approx(0.39)


def test_sync_fusion_p_eff_skips_without_fusion():
    metrics = {"fusion_p_call": 0.61, "fusion_p_eff": 0.40}
    sync_fusion_p_eff_for_direction(metrics, "CALL")
    assert metrics["fusion_p_eff"] == pytest.approx(0.40)


def test_sync_fusion_p_eff_skips_invalid_side_and_bad_values():
    metrics = {"fusion_applied": True, "fusion_p_call": "x", "fusion_p_eff": 0.40}
    sync_fusion_p_eff_for_direction(metrics, "HOLD")
    assert metrics["fusion_p_eff"] == pytest.approx(0.40)
    sync_fusion_p_eff_for_direction(metrics, "CALL")
    assert metrics["fusion_p_eff"] == pytest.approx(0.40)
    metrics["fusion_p_call"] = 1.5
    sync_fusion_p_eff_for_direction(metrics, "CALL")
    assert metrics["fusion_p_eff"] == pytest.approx(0.40)


def test_stamp_fusion_p_eff_in_range():
    metrics = {"fusion_applied": True, "fusion_p_eff": 0.55}
    stamp_fusion_p_eff(metrics)
    assert metrics["neg_edge_fusion_p_eff"] == pytest.approx(0.55)


def test_stamp_fusion_p_eff_skips_bad():
    metrics = {"fusion_applied": False, "fusion_p_eff": 0.55}
    stamp_fusion_p_eff(metrics)
    assert "neg_edge_fusion_p_eff" not in metrics
    metrics = {"fusion_applied": True, "fusion_p_eff": "bad"}
    stamp_fusion_p_eff(metrics)
    assert "neg_edge_fusion_p_eff" not in metrics
    metrics = {"fusion_applied": True, "fusion_p_eff": 0.0}
    stamp_fusion_p_eff(metrics)
    assert "neg_edge_fusion_p_eff" not in metrics


def test_resolve_neg_side_edge_prefers_fusion_p_of_exec_side():
    metrics = {
        "fusion_applied": True,
        "fusion_p_eff": 0.70,
        "fusion_p_call": 0.55,
        "fusion_p_put": 0.70,
        "calibrated_prob": 0.40,
    }
    edge = resolve_neg_side_edge(metrics, "CALL", 0.85)
    assert metrics["fusion_p_eff"] == pytest.approx(0.55)
    assert edge == pytest.approx((0.55 * 1.85) - 1.0)


def test_resolve_neg_side_edge_falls_back_on_bad_fusion_p():
    metrics = {
        "fusion_applied": True,
        "fusion_p_eff": 0.70,
        "fusion_p_call": "bad",
        "calibrated_prob": 0.60,
    }
    edge = resolve_neg_side_edge(metrics, "CALL", 0.85)
    assert edge == pytest.approx((0.60 * 1.85) - 1.0)
    metrics2 = {
        "fusion_applied": True,
        "fusion_p_call": None,
        "calibrated_prob": 0.60,
        "exec_direction": "CALL",
    }
    edge2 = resolve_neg_side_edge(metrics2, "HOLD", 0.85)
    assert isinstance(edge2, float)
