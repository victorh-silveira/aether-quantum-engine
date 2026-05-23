from src.application.services.llm.cluster_statarb_select import (
    _alignment_score,
    select_cluster_symbols_by_statarb,
)
from src.domain.models.trade import TradeDirection


def test_select_call_picks_most_negative_z():
    spreads = {"OTC_SPC": -2.8, "OTC_NDX": -0.4, "OTC_DJI": 1.2}
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_SPC", "OTC_NDX", "OTC_DJI"],
        TradeDirection.CALL,
        spreads,
        hmm_state=0,
        cfg={"enabled": True, "max_per_cluster": 1, "min_abs_z": 0.0},
    )
    assert picked == {"OTC_SPC"}
    assert "leader=OTC_SPC" in note


def test_select_put_picks_most_positive_z():
    spreads = {"OTC_FCHI": 0.2, "OTC_GDAXI": 3.1}
    picked, _ = select_cluster_symbols_by_statarb(
        ["OTC_FCHI", "OTC_GDAXI"],
        TradeDirection.PUT,
        spreads,
        hmm_state=0,
        cfg={"enabled": True, "max_per_cluster": 1, "min_abs_z": 0.0},
    )
    assert picked == {"OTC_GDAXI"}


def test_alignment_score_hmm_trending_and_invalid_direction():
    assert _alignment_score(-2.0, TradeDirection.CALL, 1) == 1.0
    assert _alignment_score(2.0, TradeDirection.PUT, 1) == 1.0
    assert _alignment_score(1.0, None, 0) == 0.0


def test_select_min_abs_z_fallback_picks_ranked_leader():
    spreads = {"OTC_SPC": 0.1, "OTC_NDX": 0.2}
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_SPC", "OTC_NDX"],
        TradeDirection.CALL,
        spreads,
        hmm_state=0,
        cfg={"enabled": True, "max_per_cluster": 1, "min_abs_z": 5.0},
    )
    assert len(picked) == 1
    assert "leader=" in note


def test_select_disabled_returns_all_candidates():
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_SPC", "OTC_NDX"],
        TradeDirection.CALL,
        {"OTC_SPC": -1.0, "OTC_NDX": 2.0},
        cfg={"enabled": False, "max_per_cluster": 1},
    )
    assert picked == {"OTC_SPC", "OTC_NDX"}
    assert note == "STATARB_INDEX_ALL"
