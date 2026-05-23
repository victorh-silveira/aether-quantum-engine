from src.application.services.llm.cluster_direction import cluster_direction_from_tag
from src.domain.models.trade import TradeDirection


def test_cluster_direction_from_tag_call_put():
    direction, inverted = cluster_direction_from_tag("CALL")
    assert direction == TradeDirection.CALL
    assert inverted is False
    put_dir, inverted2 = cluster_direction_from_tag("PUT")
    assert put_dir == TradeDirection.PUT
    assert inverted2 is False


def test_cluster_direction_from_tag_invalid():
    assert cluster_direction_from_tag("WAIT") == (None, False)
    assert cluster_direction_from_tag(None) == (None, False)
