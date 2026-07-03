from src.application.services.execution_quality_asymmetric_gate import (
    validate_micro_noise_gate,
    validate_recovery_asymmetric_gate,
)
from src.application.services.execution_quality_gate import apply_quality_penalty_to_metrics


def test_validate_recovery_asymmetric_gate_skips_neutral_low_conviction_in_recovery():
    metrics = {
        "universal_regime": "NEUTRO",
        "trade_score": 0.6300,
        "conviction": 0.6300,
        "val_accuracy": 0.70,
        "edge": 0.08,
        "direction_margin": 0.10,
    }
    assert validate_recovery_asymmetric_gate(metrics)
    assert metrics["regime_skip_cycle"] is True
    assert metrics["gate_reason"] == "low_conviction_neutral_skip"
    assert metrics["recovery_asymmetric_gate"] is True


def test_validate_recovery_asymmetric_gate_allows_neutral_above_floor():
    metrics = {
        "universal_regime": "NEUTRO",
        "trade_score": 0.70,
        "val_accuracy": 0.70,
        "edge": 0.08,
    }
    assert not validate_recovery_asymmetric_gate(metrics)
    assert not metrics.get("regime_skip_cycle")


def test_validate_recovery_asymmetric_gate_ignores_classified_regime():
    metrics = {
        "universal_regime": "COMPRESSION_TRAP",
        "trade_score": 0.50,
        "val_accuracy": 0.70,
        "edge": 0.08,
    }
    assert not validate_recovery_asymmetric_gate(metrics)


def test_validate_micro_noise_gate_skips_collapsed_adx_chop():
    metrics = {"indicators": {"adx": 0.10, "bb_width": 0.05, "hurst": 0.55}}
    assert validate_micro_noise_gate(metrics)
    assert metrics["regime_skip_cycle"] is True
    assert metrics["gate_reason"] == "micro_adx_chop_skip"
    assert metrics["micro_noise_gate"] is True


def test_validate_micro_noise_gate_skips_squeeze_random_walk():
    metrics = {"indicators": {"adx": 0.30, "bb_width": 0.005, "hurst": 0.44}}
    assert validate_micro_noise_gate(metrics)
    assert metrics["regime_skip_cycle"] is True
    assert metrics["gate_reason"] == "micro_squeeze_breakout_skip"
    assert metrics["micro_noise_gate"] is True


def test_validate_micro_noise_gate_allows_clean_trend():
    metrics = {"indicators": {"adx": 0.30, "bb_width": 0.05, "hurst": 0.58}}
    assert not validate_micro_noise_gate(metrics)
    assert not metrics.get("regime_skip_cycle")


def test_validate_micro_noise_gate_allows_squeeze_with_persistent_hurst():
    metrics = {"indicators": {"adx": 0.30, "bb_width": 0.005, "hurst": 0.55}}
    assert not validate_micro_noise_gate(metrics)
    assert not metrics.get("regime_skip_cycle")


def test_validate_micro_noise_gate_defaults_are_safe_without_indicators():
    metrics = {}
    assert not validate_micro_noise_gate(metrics)
    assert not metrics.get("regime_skip_cycle")


def test_apply_quality_penalty_tags_neutro_when_regime_unclassified():
    metrics = {
        "dl_direction": "CALL",
        "exec_direction": "CALL",
        "trade_score": 0.75,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "indicators": {"adx": 0.21, "hurst": 0.51, "vol_ratio": 0.90, "rsi": 0.50, "cmo": 0.05},
    }
    apply_quality_penalty_to_metrics(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
    )
    assert metrics.get("universal_regime") == "NEUTRO"
