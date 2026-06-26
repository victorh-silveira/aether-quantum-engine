from src.application.services.deep_learning.dl_gating import direction_from_raw_prob, resolve_confidence_thresholds
from src.domain.models.trade import TradeDirection


def test_confidence_threshold_defaults():
    call_thr, put_thr = resolve_confidence_thresholds({})
    assert call_thr == 0.75
    assert put_thr == 0.25


def test_gray_zone_not_mapped_to_direction():
    call_thr, put_thr = resolve_confidence_thresholds(
        {"confidence_call_threshold": 0.55, "confidence_put_threshold": 0.45}
    )
    assert direction_from_raw_prob(0.52, call_threshold=call_thr, put_threshold=put_thr) is None


def test_strong_call_maps_to_direction():
    assert direction_from_raw_prob(0.80) == TradeDirection.CALL
