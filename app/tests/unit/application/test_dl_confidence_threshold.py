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


def test_neutral_zone_returns_none():
    cal, resolved, mode = apply_calibration_neutral_tolerance(
        calibrated_prob=0.52,
        raw_prob=0.52,
        direction=None,
        pivot=0.5,
        neutral_lo=0.48,
        neutral_hi=0.52,
    )
    assert resolved is None
    assert mode == "neutral_zone"
