import pytest

from src.application.services.bb_width_adaptive_squeeze import (
    BB_WIDTH_HARMONIC_WINDOW,
    record_bb_width,
    reset_bb_width_buffer,
)
from src.application.services.deep_learning.dl_params import parse_indicator_gating_config
from src.application.services.execution_sniper_gates import (
    apply_bb_squeeze_requirement,
    apply_hurst_noise_veto,
    hurst_regime_allowed,
    resolve_calibration_neutral_band,
)


def test_resolve_calibration_neutral_band_from_drift():
    lo, hi = resolve_calibration_neutral_band({"calibration_neutral_drift": [0.42, 0.58]})
    assert lo == pytest.approx(0.42)
    assert hi == pytest.approx(0.58)


def test_hurst_noise_band_vetoed():
    cfg = {
        "enabled": True,
        "veto_on_noise": True,
        "noise_hurst_lo": 0.45,
        "noise_hurst_hi": 0.55,
        "strong_trend_min": 0.65,
        "strong_revert_max": 0.35,
    }
    assert hurst_regime_allowed(0.50, cfg) is False
    assert hurst_regime_allowed(0.70, cfg) is True
    assert hurst_regime_allowed(0.30, cfg) is True
    assert hurst_regime_allowed(0.60, cfg) is True


def test_apply_hurst_noise_veto_sets_gate_reason():
    cfg = parse_indicator_gating_config(
        {
            "indicator_gating": {
                "enabled": True,
                "veto_on_noise": True,
                "noise_hurst_lo": 0.45,
                "noise_hurst_hi": 0.55,
                "strong_trend_min": 0.65,
                "strong_revert_max": 0.35,
            }
        }
    )
    metrics = {"indicators": {"hurst": 0.51}, "trade_score": 0.8}
    assert apply_hurst_noise_veto(metrics, cfg) is False
    assert metrics.get("gate_reason") is None


def test_bb_squeeze_requirement_blocks_without_compression():
    reset_bb_width_buffer()
    for _ in range(BB_WIDTH_HARMONIC_WINDOW):
        record_bb_width(0.08)
    metrics = {
        "indicators": {"bb_width": 0.07},
        "bb_width_anomaly_ratio": 0.40,
        "trade_score": 0.8,
    }
    assert (
        apply_bb_squeeze_requirement(
            metrics,
            {"enabled": True, "require_extreme_squeeze": True, "anomaly_ratio": 0.40},
        )
        is False
    )
    assert metrics.get("gate_reason") is None


def test_bb_squeeze_requirement_allows_extreme_compression():
    reset_bb_width_buffer()
    for _ in range(BB_WIDTH_HARMONIC_WINDOW):
        record_bb_width(0.10)
    metrics = {
        "indicators": {"bb_width": 0.03},
        "trade_score": 0.8,
    }
    assert (
        apply_bb_squeeze_requirement(
            metrics,
            {"enabled": True, "require_extreme_squeeze": True, "anomaly_ratio": 0.40},
        )
        is False
    )
    assert metrics.get("gate_reason") is None


def test_hurst_missing_vetoed_when_configured():
    cfg = {"enabled": True, "veto_missing_hurst": True}
    assert hurst_regime_allowed(None, cfg) is False
    assert apply_hurst_noise_veto({"trade_score": 0.7}, cfg) is False


def test_hurst_static_band_without_noise_veto():
    cfg = {"enabled": True, "veto_on_noise": False, "hurst_min": 0.40, "hurst_max": 0.70}
    assert hurst_regime_allowed(0.55, cfg) is True
    assert hurst_regime_allowed(0.20, cfg) is False


def test_bb_squeeze_requirement_noop_without_extreme_flag():
    metrics = {"indicators": {"bb_width": 0.09}, "trade_score": 0.8}
    assert apply_bb_squeeze_requirement(metrics, {"enabled": True, "require_extreme_squeeze": False}) is False
    assert metrics.get("gate_reason") is None
