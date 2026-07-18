from types import SimpleNamespace
from unittest.mock import patch

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_quality_gate import (
    apply_quality_penalty_to_metrics,
    passes_execution_quality,
    quality_gate_params,
)
from src.application.services.execution_quality_gate_cluster import quality_conviction_suspends_cluster
from src.application.services.execution_quality_gate_fallback import (
    _hard_quality_reject_for_fallback,
    cluster_quality_gate_blocks_mandatory_fallback,
)
from src.application.services.execution_quality_gate_microstructure import (
    apply_microstructure_starvation_veto,
    is_microstructure_starvation_reason,
    resolve_min_adx_threshold,
)
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


def test_quality_gate_params_reads_min_adx_threshold_alias():
    params = quality_gate_params({"quality_gate": {"min_adx_threshold": 0.20}})
    assert params["min_adx_normal"] == 0.20


def test_passes_execution_quality_rejects_low_adx():
    metrics = _healthy_metrics(indicators={"adx": 0.13, "vol_ratio": 1.0})
    exec_cfg = {"quality_gate": {"min_adx_threshold": 0.20}}
    orch = SimpleNamespace(
        config={
            "deep_learning": {"indicator_gating": {"enabled": True, "vol_ratio_min": 0.65}},
            "risk_management": {"min_validation_accuracy_gate": 0.63},
        }
    )
    assert passes_execution_quality(metrics, exec_cfg=exec_cfg, orch=orch) is False
    assert metrics["quality_gate_reason"] == "adx_starvation"
    assert metrics["regime_skip_cycle"] is True


def test_passes_execution_quality_rejects_low_vol_ratio():
    metrics = _healthy_metrics(indicators={"adx": 0.25, "vol_ratio": 0.50})
    exec_cfg = {"quality_gate": {"min_adx_threshold": 0.20}}
    orch = SimpleNamespace(
        config={
            "deep_learning": {"indicator_gating": {"enabled": True, "vol_ratio_min": 0.65}},
            "risk_management": {"min_validation_accuracy_gate": 0.63},
        }
    )
    assert passes_execution_quality(metrics, exec_cfg=exec_cfg, orch=orch) is False
    assert metrics["quality_gate_reason"] == "vol_ratio_starvation"


def test_passes_execution_quality_rejects_low_val_accuracy():
    metrics = _healthy_metrics(val_accuracy=0.59)
    exec_cfg = {"quality_gate": {"min_adx_threshold": 0.20}}
    orch = SimpleNamespace(
        config={
            "deep_learning": {"indicator_gating": {"enabled": True, "vol_ratio_min": 0.65}},
            "risk_management": {"min_validation_accuracy_gate": 0.63},
        }
    )
    assert passes_execution_quality(metrics, exec_cfg=exec_cfg, orch=orch) is False
    assert metrics["quality_gate_reason"] == "val_accuracy_gate"


def test_passes_execution_quality_rejects_healthy_microstructure_with_weak_margin():
    metrics = _healthy_metrics(direction_margin=0.01, calibrated_prob=0.51)
    exec_cfg = {
        "quality_gate": {
            "min_adx_threshold": 0.20,
            "min_direction_margin": 0.12,
        }
    }
    orch = SimpleNamespace(
        config={
            "deep_learning": {"indicator_gating": {"enabled": True, "vol_ratio_min": 0.65}},
            "risk_management": {"min_validation_accuracy_gate": 0.63},
        }
    )
    assert passes_execution_quality(metrics, exec_cfg=exec_cfg, orch=orch) is False
    assert metrics.get("quality_gate_reason") == "direction_margin_gate"
    assert metrics.get("quality_guard_reject") is True


def test_quality_conviction_suspends_cluster_on_microstructure_starvation(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = True
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["quality_gate"] = {"min_adx_threshold": 0.20}
    orch.config.setdefault("deep_learning", {})["indicator_gating"] = {
        "enabled": True,
        "vol_ratio_min": 0.65,
        "adx_min": 0.20,
    }
    orch.config.setdefault("risk_management", {})["min_validation_accuracy_gate"] = 0.63
    decisions = {
        "RDBEAR": {
            "metrics": _healthy_metrics(
                indicators={"adx": 0.13, "vol_ratio": 0.21},
                val_accuracy=0.59,
            )
        }
    }
    assert quality_conviction_suspends_cluster(orch, decisions) is True
    assert is_microstructure_starvation_reason(decisions["RDBEAR"]["metrics"]["quality_gate_reason"])


def test_mandatory_fallback_blocked_by_microstructure_starvation():
    decisions = {
        "RDBEAR": {
            "direction": "PUT",
            "metrics": {
                "deploy_ok": True,
                "calibrated_prob": 0.55,
                "quality_gate_reason": "vol_ratio_starvation",
                "quality_guard_reject": True,
            },
        }
    }
    assert (
        cluster_quality_gate_blocks_mandatory_fallback(
            decisions,
            exec_cfg={},
            risk_manager=SimpleNamespace(consecutive_losses_linear=0, pending_loss_total=lambda: 0.0),
            trade_symbols=["RDBEAR"],
        )
        is True
    )


def test_apply_quality_penalty_unit_on_starvation():
    metrics = _healthy_metrics(indicators={"adx": 0.10, "vol_ratio": 1.0})
    penalty = apply_quality_penalty_to_metrics(
        metrics,
        exec_cfg={"quality_gate": {"min_adx_threshold": 0.20}},
        risk_manager=None,
    )
    assert penalty == 1.0


def test_apply_microstructure_starvation_veto_none_when_thresholds_disabled():
    metrics = _healthy_metrics(indicators={"adx": 0.05, "vol_ratio": 0.10}, val_accuracy=0.40)
    assert apply_microstructure_starvation_veto(metrics, exec_cfg={}, orch=None, risk_manager=None) is None


def test_resolve_min_adx_threshold_falls_back_to_min_adx_normal():
    assert resolve_min_adx_threshold({"quality_gate": {"min_adx_normal": 0.22}}) == 0.22
    assert resolve_min_adx_threshold({"quality_gate": "bad"}) == 0.0


def test_indicator_float_reads_macro_micro_and_top_level():
    orch = SimpleNamespace(
        config={
            "deep_learning": {"indicator_gating": {"enabled": True, "vol_ratio_min": 0.65}},
            "risk_management": {"min_validation_accuracy_gate": 0.63},
        }
    )
    exec_cfg = {"quality_gate": {"min_adx_threshold": 0.20}}
    assert (
        apply_microstructure_starvation_veto(
            {"macro_indicators": {"adx": "bad"}, "val_accuracy": 0.70},
            exec_cfg=exec_cfg,
            orch=orch,
        )
        is None
    )
    assert (
        apply_microstructure_starvation_veto(
            {"adx": "bad", "val_accuracy": 0.70, "vol_ratio": 1.0},
            exec_cfg=exec_cfg,
            orch=orch,
        )
        is None
    )
    assert (
        apply_microstructure_starvation_veto(
            {"adx": 0.10, "val_accuracy": 0.70, "vol_ratio": 1.0},
            exec_cfg=exec_cfg,
            orch=orch,
        )
        == "adx_starvation"
    )
    assert (
        apply_microstructure_starvation_veto(
            {
                "indicators": {"adx": 0.25},
                "vol_ratio_short_long": 0.40,
                "val_accuracy": 0.70,
            },
            exec_cfg=exec_cfg,
            orch=orch,
        )
        == "vol_ratio_starvation"
    )
    assert (
        apply_microstructure_starvation_veto(
            {
                "indicators": {"adx": 0.25, "vol_ratio": 1.0},
                "val_accuracy": "bad",
            },
            exec_cfg=exec_cfg,
            orch=orch,
        )
        is None
    )


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
            symbol="RDBEAR",
            orch=orch,
        )
    assert result is None
    assert entry["metrics"]["quality_gate_reason"] == "adx_starvation"


def test_hard_quality_reject_for_fallback_microstructure_reason():
    assert _hard_quality_reject_for_fallback({"quality_gate_reason": "adx_starvation"}) is True
