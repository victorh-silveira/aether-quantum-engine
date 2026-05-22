from unittest.mock import MagicMock

from src.application.services.llm.llm_cluster_conviction import (
    cluster_execution_direction,
    cluster_follow_conviction_threshold,
    invert_cluster_tag,
)
from src.domain.models.trade import TradeDirection


def test_cluster_follow_threshold_from_config():
    orch = MagicMock()
    orch.config = {
        "llm": {
            "indicator_config": {"cluster_follow_conviction_threshold": 0.82},
            "min_conviction_execute": 0.60,
        }
    }
    assert cluster_follow_conviction_threshold(orch) == 0.82


def test_cluster_follow_threshold_fallbacks():
    orch = MagicMock()
    orch.config = {"llm": {"indicator_config": {"follow_threshold": 0.80}, "min_conviction_execute": 0.70}}
    assert cluster_follow_conviction_threshold(orch) == 0.80
    orch.config = {"llm": {"min_conviction_execute": 0.75}}
    assert cluster_follow_conviction_threshold(orch) == 0.75


def test_cluster_execution_direction_follow_and_inverse():
    follow_dir, inv = cluster_execution_direction("CALL", 0.90, 0.85)
    assert follow_dir == TradeDirection.CALL
    assert inv is False
    put_dir, inv2 = cluster_execution_direction("CALL", 0.60, 0.85)
    assert put_dir == TradeDirection.PUT
    assert inv2 is True
    assert cluster_execution_direction("WAIT", 0.5, 0.85) == (None, False)
    assert invert_cluster_tag("CALL") == "PUT"
    assert invert_cluster_tag("PUT") == "CALL"
    assert invert_cluster_tag(None) is None
