from src.application.services.deep_learning.dl_gating import direction_from_raw_prob, resolve_confidence_thresholds
from src.domain.models.trade import TradeDirection


def test_direction_from_raw_prob_uses_thresholds():
    call_thr, put_thr = resolve_confidence_thresholds({})
    assert direction_from_raw_prob(0.80, call_threshold=call_thr, put_threshold=put_thr) == TradeDirection.CALL


def test_direction_from_raw_prob_put_threshold():
    assert direction_from_raw_prob(0.20, call_threshold=0.75, put_threshold=0.25) == TradeDirection.PUT
