from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_quality_gate import passes_execution_quality
from src.application.services.execution_quality_gate_cluster import quality_conviction_suspends_cluster
from src.application.services.execution_quality_gate_fallback import _hard_quality_reject_for_fallback
from src.application.services.execution_quality_gate_microstructure import apply_microstructure_starvation_veto
from src.domain.models.trade import TradeDirection


def _healthy_metrics(**overrides) -> dict:
    base = {
        "deploy_ok": True,
        "calibrated_prob": 0.62,
        "val_accuracy": 0.68,
        "direction_margin": 0.02,
        "indicators": {"adx": 0.25, "vol_ratio": 1.0},
    }
    base.update(overrides)
    return base


def test_resolve_execution_direction_aborts_on_adx_starvation():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "deploy_ok": True,
            "execute": True,
            "raw_prob": 0.35,
            "calibrated_prob": 0.35,
            "val_accuracy": 0.70,
            "predicted_payoff_edge": 0.20,
            "meta_classifier_applied": True,
            "indicators": {"adx": 0.12, "vol_ratio": 1.0},
        },
    }
    orch = SimpleNamespace(
        config={
            "deep_learning": {"indicator_gating": {"enabled": False}},
            "risk_management": {"min_validation_accuracy_gate": 0.0},
        }
    )
    with patch(
        "src.application.services.execution_direction_resolver.resolve_meta_payoff_edge",
        return_value=(0.20, True),
    ):
        result = resolve_execution_direction(
            entry,
            exec_cfg={"quality_gate": {"min_adx_threshold": 0.20}},
            symbol="R_10",
            orch=orch,
        )
    assert result is None
    assert entry["metrics"]["quality_gate_reason"] == "adx_starvation"
    assert entry["metrics"]["gate_reason"] == "adx_starvation"


def test_hard_quality_reject_for_fallback_microstructure_reason():
    assert _hard_quality_reject_for_fallback({"quality_gate_reason": "adx_starvation"}) is True


def test_adx_starvation_decays_with_skipped_cycles():
    metrics_low = {"adx": 0.10, "val_accuracy": 0.70, "vol_ratio": 1.0}
    exec_cfg = {"quality_gate": {"min_adx_threshold": 0.20}}
    assert (
        apply_microstructure_starvation_veto(
            metrics_low,
            exec_cfg=exec_cfg,
            skipped_cycles_counter=0,
        )
        == "adx_starvation"
    )
    assert "quality_adx_detail" in metrics_low
    metrics_escape = {"adx": 0.10, "val_accuracy": 0.70, "vol_ratio": 1.0}
    assert (
        apply_microstructure_starvation_veto(
            metrics_escape,
            exec_cfg=exec_cfg,
            skipped_cycles_counter=9,
        )
        is None
    )
    assert metrics_escape["quality_min_adx"] == pytest.approx(0.08)
    assert metrics_escape["quality_adx_decay_factor"] == pytest.approx(0.40)


def test_passes_execution_quality_allows_low_adx_after_starvation_decay():
    metrics = _healthy_metrics(
        indicators={"adx": 0.10, "vol_ratio": 1.0},
        direction_margin=0.20,
        calibrated_prob=0.70,
    )
    exec_cfg = {
        "quality_gate": {
            "min_adx_threshold": 0.20,
            "min_direction_margin": 0.08,
            "regular": {"min_direction_margin": 0.08, "min_payoff_edge": -999.0},
        }
    }
    orch = SimpleNamespace(
        config={
            "deep_learning": {"indicator_gating": {"enabled": False}},
            "risk_management": {"min_validation_accuracy_gate": 0.0},
        },
        _quality_skipped_cycles_counter=9,
    )
    assert passes_execution_quality(metrics, exec_cfg=exec_cfg, orch=orch, skipped_cycles_counter=9) is True
    assert metrics.get("quality_gate_reason") is None


def test_vol_ratio_starvation_decays_with_skipped_cycles():
    orch = SimpleNamespace(
        config={
            "deep_learning": {"indicator_gating": {"enabled": True, "vol_ratio_min": 0.65}},
            "risk_management": {"min_validation_accuracy_gate": 0.0},
        }
    )
    exec_cfg = {"quality_gate": {"min_adx_threshold": 0.0}}
    metrics_low = {"indicators": {"adx": 0.30, "vol_ratio": 0.20}, "val_accuracy": 0.70}
    assert (
        apply_microstructure_starvation_veto(
            metrics_low,
            exec_cfg=exec_cfg,
            orch=orch,
            skipped_cycles_counter=0,
        )
        == "vol_ratio_starvation"
    )
    assert "quality_vol_ratio_detail" in metrics_low
    metrics_escape = {"indicators": {"adx": 0.30, "vol_ratio": 0.20}, "val_accuracy": 0.70}
    assert (
        apply_microstructure_starvation_veto(
            metrics_escape,
            exec_cfg=exec_cfg,
            orch=orch,
            skipped_cycles_counter=11,
        )
        is None
    )
    assert metrics_escape["quality_min_vol_ratio"] == pytest.approx(0.065)


def test_quality_conviction_includes_adx_and_vol_detail_in_log(orch_ready, caplog):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["quality_gate"] = {"min_adx_threshold": 0.20}
    orch.config.setdefault("deep_learning", {})["indicator_gating"] = {
        "enabled": True,
        "vol_ratio_min": 0.65,
        "adx_min": 0.20,
    }
    orch.config.setdefault("risk_management", {})["min_validation_accuracy_gate"] = 0.0
    orch._active_cycle_id = 901
    decisions_adx = {
        "skip": "invalid",
        "bad_metrics": {"metrics": "x"},
        "R_10": {
            "metrics": {
                "deploy_ok": True,
                "calibrated_prob": 0.62,
                "val_accuracy": 0.70,
                "indicators": {"adx": 0.10, "vol_ratio": 1.0},
            }
        },
    }
    with caplog.at_level("INFO", logger="AETH"):
        assert quality_conviction_suspends_cluster(orch, decisions_adx) is True
    assert any("adx=" in r.message for r in caplog.records)
    caplog.clear()
    orch._active_cycle_id = 902
    decisions_vol = {
        "R_10": {
            "metrics": {
                "deploy_ok": True,
                "calibrated_prob": 0.62,
                "val_accuracy": 0.70,
                "indicators": {"adx": 0.30, "vol_ratio": 0.20},
            }
        }
    }
    with caplog.at_level("INFO", logger="AETH"):
        assert quality_conviction_suspends_cluster(orch, decisions_vol) is True
    assert any("vol_ratio=" in r.message for r in caplog.records)
