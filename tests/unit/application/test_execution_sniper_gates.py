"""Helpers de calibracao neutra apos remocao do veto Hurst."""

import pytest

from src.application.services.execution_sniper_gates import resolve_calibration_neutral_band


def test_resolve_calibration_neutral_band_from_half_width():
    lo, hi = resolve_calibration_neutral_band({"neutral_half_width": 0.04})
    assert lo == 0.46
    assert hi == 0.54


def test_resolve_calibration_neutral_band_from_drift():
    lo, hi = resolve_calibration_neutral_band({"calibration_neutral_drift": [0.48, 0.52]})
    assert lo == 0.48
    assert hi == 0.52


def test_resolve_calibration_neutral_band_rejects_degenerate_drift():
    lo, hi = resolve_calibration_neutral_band({"calibration_neutral_drift": [0.5, 0.5], "neutral_half_width": 0.03})
    assert lo == pytest.approx(0.47)
    assert hi == pytest.approx(0.53)


def test_resolve_calibration_neutral_band_prefers_calibration_half_width():
    lo, hi = resolve_calibration_neutral_band(
        {"calibration_neutral_drift": [0.5, 0.5], "neutral_calibration_half_width": 0.03}
    )
    assert lo == pytest.approx(0.47)
    assert hi == pytest.approx(0.53)
