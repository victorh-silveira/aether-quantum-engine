import pytest

from src.application.services.execution_sniper_gates import (
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


def test_hurst_missing_vetoed_when_configured():
    cfg = {"enabled": True, "veto_missing_hurst": True}
    assert hurst_regime_allowed(None, cfg) is False


def test_hurst_static_band_without_noise_veto():
    cfg = {"enabled": True, "veto_on_noise": False, "hurst_min": 0.40, "hurst_max": 0.70}
    assert hurst_regime_allowed(0.55, cfg) is True
    assert hurst_regime_allowed(0.20, cfg) is False


def test_resolve_calibration_neutral_band_from_half_width():
    lo, hi = resolve_calibration_neutral_band({"neutral_half_width": 0.05})
    assert lo == pytest.approx(0.45)
    assert hi == pytest.approx(0.55)
