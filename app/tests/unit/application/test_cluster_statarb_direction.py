from src.application.services.llm.cluster_statarb_direction import direction_from_statarb_z
from src.domain.models.trade import TradeDirection


def test_direction_from_statarb_z_mean_reversion():
    assert direction_from_statarb_z(2.8, hmm_state=0, z_threshold=2.5, min_abs_z=0.65) == TradeDirection.PUT
    assert direction_from_statarb_z(-2.8, hmm_state=0, z_threshold=2.5, min_abs_z=0.65) == TradeDirection.CALL
    assert direction_from_statarb_z(1.0, hmm_state=0, z_threshold=2.5, min_abs_z=0.65) is None
    assert direction_from_statarb_z(0.2, hmm_state=0, z_threshold=2.5, min_abs_z=0.65) is None


def test_direction_from_statarb_z_trending():
    assert direction_from_statarb_z(0.70, hmm_state=1, z_threshold=2.5, min_abs_z=0.65) == TradeDirection.CALL
    assert direction_from_statarb_z(-0.70, hmm_state=1, z_threshold=2.5, min_abs_z=0.65) == TradeDirection.PUT
