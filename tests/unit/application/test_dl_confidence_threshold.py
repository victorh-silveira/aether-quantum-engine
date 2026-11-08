from src.application.services.deep_learning.dl_calibration_tolerance import (
    apply_calibration_neutral_tolerance,
    infer_direction_from_prob as direction_from_raw_prob,
)
from src.application.services.deep_learning.dl_gating import resolve_confidence_thresholds
from src.domain.models.trade import TradeDirection


def test_confidence_threshold_defaults():
    call_thr, put_thr = resolve_confidence_thresholds({})
    assert call_thr == 0.55
    assert put_thr == 0.45


def test_direction_call_above_pivot():
    assert direction_from_raw_prob(0.80, None) == TradeDirection.CALL


def test_direction_put_below_pivot():
    assert direction_from_raw_prob(0.20, None) == TradeDirection.PUT


def test_direction_preserves_explicit():
    assert direction_from_raw_prob(0.50, TradeDirection.CALL) == TradeDirection.CALL


def test_mid_cal_resolves_call_vs_half():
    cal, resolved, mode = apply_calibration_neutral_tolerance(
        calibrated_prob=0.50,
        raw_prob=0.50,
        direction=None,
        pivot=0.5,
        neutral_lo=0.48,
        neutral_hi=0.52,
    )
    assert cal == 0.50
    assert resolved == TradeDirection.CALL
    assert mode == "calibrated"
