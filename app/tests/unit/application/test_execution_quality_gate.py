import pytest

from src.application.services.execution_quality_gate import (
    MANDATORY_MIN_TRADE_SCORE_DEFAULT,
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


def test_quality_gate_params_defaults():
    params = quality_gate_params({})
    assert params["min_direction_margin"] == 0.06
    assert params["inverted_min_score"] == 0.76
    assert params["min_adx_normal"] == 0.20
    assert pytest.approx(0.72) == MANDATORY_MIN_TRADE_SCORE_DEFAULT


def test_passes_execution_quality_rejects_low_edge():
    metrics = {"trade_score": 0.80, "val_accuracy": 0.70, "edge": 0.02, "direction_margin": 0.08}
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)


def test_passes_execution_quality_uses_calibrated_edge():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "calibrated_prob": 0.52,
        "calibrated_edge": 0.02,
        "direction_margin": 0.08,
    }
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)


def test_passes_execution_quality_uses_dynamic_min_edge():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.05,
        "calibrated_edge": 0.05,
        "dynamic_min_edge": 0.06,
        "direction_margin": 0.08,
    }
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)


def test_passes_execution_quality_rejects_low_val():
    metrics = {"trade_score": 0.80, "val_accuracy": 0.50, "edge": 0.10, "direction_margin": 0.08}
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)


def test_passes_execution_quality_accepts_complete_signal():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "indicators": {"adx": 0.25},
    }
    assert passes_execution_quality(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
        min_adx_normal=0.18,
        recovery_active=False,
    )


def test_passes_execution_quality_rejects_low_signal():
    metrics = {"trade_score": 0.55, "val_accuracy": 0.70, "edge": 0.10, "direction_margin": 0.08}
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)


def test_passes_execution_quality_rejects_low_margin():
    metrics = {"trade_score": 0.80, "val_accuracy": 0.70, "edge": 0.10, "direction_margin": 0.02}
    assert not passes_execution_quality(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
        min_direction_margin=0.05,
    )


def test_passes_execution_quality_rejects_inverted_without_strong_signal():
    metrics = {
        "trade_score": 0.70,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "direction_inverted": True,
    }
    assert not passes_execution_quality(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
        inverted_min_score=0.74,
    )


def test_passes_execution_quality_accepts_strong_inverted():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.15,
        "direction_margin": 0.10,
        "direction_inverted": True,
    }
    assert passes_execution_quality(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
        inverted_min_score=0.74,
    )


def test_passes_execution_quality_rejects_low_adx_in_normal_mode():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "indicators": {"adx": 0.12},
    }
    assert not passes_execution_quality(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
        min_adx_normal=0.18,
        recovery_active=False,
    )


def test_passes_execution_quality_skips_adx_check_in_recovery():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "indicators": {"adx": 0.12},
    }
    assert passes_execution_quality(
        metrics,
        min_signal=0.64,
        min_val=0.60,
        min_edge=0.04,
        min_adx_normal=0.18,
        recovery_active=True,
    )


def test_passes_execution_quality_rejects_exhaustion_conflict():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "exhaustion_conflict": True,
        "exhaustion_penalty": 0.15,
        "indicators": {"adx": 0.25},
    }
    assert not passes_execution_quality(
        metrics,
        min_signal=0.64,
        min_val=0.60,
        min_edge=0.04,
        exhaustion_gate_cfg={"min_penalty_skip": 0.12},
    )


def test_passes_execution_quality_rejects_hard_gate_penalty():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "exhaustion_conflict": True,
        "exhaustion_penalty": 0.37,
        "indicators": {"adx": 0.25},
    }
    assert not passes_execution_quality(
        metrics,
        min_signal=0.64,
        min_val=0.60,
        min_edge=0.04,
        exhaustion_gate_cfg={"min_penalty_skip": 0.12},
    )


def test_passes_execution_quality_recovery_uses_hurst_floor():
    metrics = {
        "trade_score": 0.64,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "indicators": {"adx": 0.25, "hurst": 0.40},
    }
    kelly = {
        "recovery_min_trade_score": 0.64,
        "recovery_hurst_persistence_min": 0.58,
        "recovery_hurst_log_scale": 0.08,
    }
    assert not passes_execution_quality(
        metrics,
        min_signal=0.64,
        min_val=0.60,
        min_edge=0.04,
        recovery_active=True,
        recovery_kelly_cfg=kelly,
        consecutive_losses=2,
    )


def test_passes_execution_quality_recovery_decay_counter_relaxes_floor():
    metrics = {
        "trade_score": 0.64,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "indicators": {"adx": 0.25, "hurst": 0.55},
    }
    kelly = {
        "recovery_min_trade_score": 0.64,
        "recovery_hurst_persistence_min": 0.58,
        "recovery_hurst_log_scale": 0.08,
        "recovery_hurst_decay_enabled": True,
        "recovery_hurst_decay_per_skip": 0.01,
        "recovery_hurst_decay_floor": 0.50,
    }
    assert passes_execution_quality(
        metrics,
        min_signal=0.64,
        min_val=0.60,
        min_edge=0.04,
        recovery_active=True,
        recovery_kelly_cfg=kelly,
        consecutive_losses=2,
        recovery_skip_counter=8,
    )


def test_passes_execution_quality_log_decay_with_severe_drawdown():
    metrics = {
        "trade_score": 0.64,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "indicators": {"adx": 0.25, "hurst": 0.54},
    }
    kelly = {
        "recovery_min_trade_score": 0.64,
        "recovery_hurst_persistence_min": 0.58,
        "recovery_hurst_log_scale": 0.08,
        "recovery_hurst_decay_enabled": True,
        "recovery_hurst_log_decay_coef": 0.025,
        "recovery_hurst_accel_losses_min": 3,
        "recovery_hurst_severe_drawdown_min": 150.0,
    }
    assert passes_execution_quality(
        metrics,
        min_signal=0.64,
        min_val=0.60,
        min_edge=0.04,
        recovery_active=True,
        recovery_kelly_cfg=kelly,
        consecutive_losses=3,
        recovery_skip_counter=6,
        session_drawdown=200.0,
    )


def test_passes_execution_quality_rejects_c0011_like_vol_compression_edge():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.05,
        "calibrated_edge": 0.05,
        "dynamic_min_edge": 0.10,
        "direction_margin": 0.08,
        "indicators": {"adx": 0.19, "vol_ratio": 0.41},
    }
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)
