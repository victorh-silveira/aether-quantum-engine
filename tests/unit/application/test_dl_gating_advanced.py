from src.application.services.deep_learning.dl_gating import direction_from_raw_prob, resolve_edge
from src.domain.models.trade import TradeDirection


def test_direction_from_raw_prob_at_call_threshold():
    assert direction_from_raw_prob(0.75, call_threshold=0.75, put_threshold=0.25) == TradeDirection.CALL


def test_direction_from_raw_prob_below_call_threshold():
    assert direction_from_raw_prob(0.749, call_threshold=0.75, put_threshold=0.25) is None


def test_resolve_edge_symmetric():
    assert resolve_edge(0.25) == resolve_edge(0.75)
