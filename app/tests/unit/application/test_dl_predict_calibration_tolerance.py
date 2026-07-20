import pytest

from src.application.services.deep_learning.dl_calibration_tolerance import (
    apply_calibration_neutral_tolerance,
    infer_direction_from_prob,
)
from src.domain.models.trade import TradeDirection


def test_calibration_mid_band_resolves_put_without_abstention():
    prob, direction, mode = apply_calibration_neutral_tolerance(0.49, 0.51, None)
    assert prob == pytest.approx(0.49)
    assert direction == TradeDirection.PUT
    assert mode == "calibrated"


def test_calibration_keeps_explicit_direction_in_former_neutral_band():
    prob, direction, mode = apply_calibration_neutral_tolerance(0.50, 0.50, TradeDirection.CALL)
    assert prob == pytest.approx(0.50)
    assert direction == TradeDirection.CALL
    assert mode == "calibrated"


def test_calibration_custom_band_keeps_call():
    prob, direction, mode = apply_calibration_neutral_tolerance(
        0.55,
        0.55,
        TradeDirection.CALL,
        neutral_lo=0.42,
        neutral_hi=0.58,
    )
    assert prob == pytest.approx(0.55)
    assert direction == TradeDirection.CALL
    assert mode == "calibrated"


def test_calibration_outside_wide_band_keeps_call():
    prob, direction, mode = apply_calibration_neutral_tolerance(
        0.62,
        0.60,
        TradeDirection.CALL,
        neutral_lo=0.42,
        neutral_hi=0.58,
    )
    assert prob == pytest.approx(0.62)
    assert direction == TradeDirection.CALL
    assert mode == "calibrated"


def test_calibration_tcn_macro_override_privileges_raw_call():
    prob, direction, mode = apply_calibration_neutral_tolerance(0.50, 0.70, None)
    assert prob == pytest.approx(0.70)
    assert direction == TradeDirection.CALL
    assert mode == "tcn_macro_override"


def test_calibration_tcn_macro_override_privileges_raw_put():
    prob, direction, mode = apply_calibration_neutral_tolerance(0.50, 0.30, TradeDirection.CALL)
    assert prob == pytest.approx(0.30)
    assert direction == TradeDirection.CALL
    assert mode == "tcn_macro_override"


def test_calibration_outside_neutral_infers_when_direction_missing():
    prob, direction, mode = apply_calibration_neutral_tolerance(0.60, 0.55, None)
    assert prob == pytest.approx(0.60)
    assert direction == TradeDirection.CALL
    assert mode == "calibrated"


def test_infer_direction_keeps_explicit_side():
    assert infer_direction_from_prob(0.40, TradeDirection.CALL) == TradeDirection.CALL
