import pytest

from src.application.services.execution_quality_gate import (
    MANDATORY_MIN_TRADE_SCORE_DEFAULT,
    apply_quality_penalty_to_metrics,
    passes_execution_quality,
    quality_gate_params,
)


def test_quality_gate_params_custom_config():
    params = quality_gate_params(
        {"quality_gate": {"min_direction_margin": 0.08, "inverted_min_score": 0.80, "min_adx_normal": 0.20}}
    )
    assert params["min_direction_margin"] == 0.08
    assert params["inverted_min_score"] == 0.80
    assert params["min_adx_normal"] == 0.20


def test_quality_gate_params_defaults_are_neutral():
    params = quality_gate_params({})
    assert params["min_direction_margin"] == 0.0
    assert params["inverted_min_score"] == 0.0
    assert params["min_adx_normal"] == 0.0
    assert pytest.approx(0.72) == MANDATORY_MIN_TRADE_SCORE_DEFAULT


def test_quality_gate_params_ignores_non_dict_config():
    assert quality_gate_params({"quality_gate": "invalid"})["min_direction_margin"] == 0.0
    assert quality_gate_params("nope")["min_adx_normal"] == 0.0


def test_passes_execution_quality_always_true_for_weak_signal():
    metrics = {"trade_score": 0.20, "val_accuracy": 0.10, "edge": 0.001, "direction_margin": 0.0}
    assert passes_execution_quality(metrics, min_signal=0.90, min_val=0.90, min_edge=0.50) is True
    assert metrics["regime_skip_cycle"] is False


def test_passes_execution_quality_never_skips_regardless_of_kwargs():
    metrics = {"trade_score": 0.55, "direction_inverted": True, "indicators": {"adx": 0.01}}
    assert (
        passes_execution_quality(
            metrics,
            min_signal=0.99,
            min_val=0.99,
            min_edge=0.99,
            min_direction_margin=0.5,
            inverted_min_score=0.99,
            min_adx_normal=0.99,
            recovery_active=True,
            consecutive_losses=5,
        )
        is True
    )
    assert metrics["regime_skip_cycle"] is False


def test_apply_quality_penalty_is_zero_and_never_skips():
    metrics = {"trade_score": 0.51, "conviction": 0.51}
    penalty = apply_quality_penalty_to_metrics(
        metrics,
        min_signal=0.72,
        min_val=0.60,
        min_edge=0.04,
        recovery_active=True,
    )
    assert penalty == 0.0
    assert metrics["regime_skip_cycle"] is False
    assert metrics["trade_score"] == 0.51
