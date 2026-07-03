from src.application.services.execution_quality_gate import (
    apply_quality_penalty_to_metrics,
    apply_vol_cohesion_entropic_downgrade,
    enforce_micro_middle_uncertainty_skip,
    passes_execution_quality,
)
from src.application.services.execution_universal_regime_types import (
    MICRO_MIDDLE_UNCERTAINTY_REASON,
    RegimeState,
)


def test_apply_vol_cohesion_downgrades_expanded_macro_with_compressed_micro():
    metrics = {
        "macro_indicators": {"vol_ratio": 1.30},
        "indicators": {"vol_ratio": 0.42, "adx": 0.25},
        "dl_direction": "CALL",
        "exec_direction": "CALL",
    }
    assert apply_vol_cohesion_entropic_downgrade(metrics) is True
    assert metrics["universal_regime"] == RegimeState.ENTROPIC_NOISE.value
    assert metrics["gate_penalty"] == "noise"
    assert metrics["regime_skip_cycle"] is True
    assert metrics["vol_cohesion_divergence"] is True


def test_apply_vol_cohesion_ignores_aligned_timeframes():
    metrics = {
        "macro_indicators": {"vol_ratio": 1.30},
        "indicators": {"vol_ratio": 1.10, "adx": 0.25},
    }
    assert apply_vol_cohesion_entropic_downgrade(metrics) is False
    assert "universal_regime" not in metrics


def test_passes_execution_quality_rejects_vol_cohesion_divergence():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "macro_indicators": {"vol_ratio": 1.25},
        "indicators": {"adx": 0.25, "vol_ratio": 0.45},
    }
    assert not passes_execution_quality(metrics, min_signal=0.72, min_val=0.60, min_edge=0.04)


def test_apply_quality_penalty_vol_cohesion_entropic_noise():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "dl_direction": "CALL",
        "exec_direction": "CALL",
        "macro_indicators": {"vol_ratio": 1.20},
        "indicators": {"adx": 0.25, "vol_ratio": 0.40},
    }
    apply_quality_penalty_to_metrics(
        metrics,
        min_signal=0.72,
        min_val=0.60,
        min_edge=0.04,
    )
    assert metrics.get("universal_regime") == RegimeState.ENTROPIC_NOISE.value
    assert metrics.get("vol_cohesion_divergence") is True


def test_enforce_micro_middle_uncertainty_skip_sets_regime_skip():
    metrics = {"gate_reason": MICRO_MIDDLE_UNCERTAINTY_REASON}
    assert enforce_micro_middle_uncertainty_skip(metrics) is True
    assert metrics["regime_skip_cycle"] is True
    assert metrics["micro_middle_uncertainty"] is True


def test_enforce_micro_middle_uncertainty_skip_ignores_other_reasons():
    metrics = {"gate_reason": "noise"}
    assert enforce_micro_middle_uncertainty_skip(metrics) is False
    assert "regime_skip_cycle" not in metrics


def test_passes_execution_quality_blocks_micro_middle_uncertainty():
    metrics = {
        "trade_score": 0.50,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "gate_reason": MICRO_MIDDLE_UNCERTAINTY_REASON,
    }
    assert not passes_execution_quality(metrics, min_signal=0.40, min_val=0.60, min_edge=0.04)
    assert metrics["regime_skip_cycle"] is True


def test_apply_quality_penalty_enforces_micro_middle_uncertainty():
    metrics = {
        "trade_score": 0.82,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "dl_direction": "CALL",
        "exec_direction": "CALL",
        "call_votes": 4,
        "put_votes": 2,
        "indicators": {"adx": 0.25, "hurst": 0.52, "vol_ratio": 1.0, "rsi": 0.52, "keltner": 0.50, "cmo": 0.10},
    }
    apply_quality_penalty_to_metrics(metrics, min_signal=0.72, min_val=0.60, min_edge=0.04)
    assert metrics.get("gate_reason") == MICRO_MIDDLE_UNCERTAINTY_REASON
    assert metrics.get("regime_skip_cycle") is True
