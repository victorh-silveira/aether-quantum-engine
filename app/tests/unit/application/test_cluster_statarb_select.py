from src.application.services.llm.cluster_statarb_attempt import select_cluster_symbol_attempt_order
from src.application.services.llm.cluster_statarb_select import (
    _alignment_score,
    _statarb_leader_pick,
    _wr_blend_score,
    resolve_statarb_cluster_config,
    resolve_statarb_cluster_config_for_tag,
    select_cluster_symbols_by_statarb,
    statarb_execute_min_abs_z,
    symbol_z_supports_direction,
)
from src.domain.models.trade import TradeDirection


def test_wr_blend_score_missing_symbol():
    assert _wr_blend_score("OTC_X", {"OTC_FCHI": 0.5}, 0.4) == 0.0


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
    assert _alignment_score(2.0, TradeDirection.CALL, 1) == 1.0
    assert _alignment_score(-2.0, TradeDirection.PUT, 1) == 1.0
    assert _alignment_score(-2.0, TradeDirection.CALL, 1) == 0.0
    assert _alignment_score(1.0, None, 0) == 0.0


def test_statarb_leader_pick_includes_wr_in_note():
    picked, note = _statarb_leader_pick(
        [("OTC_GDAXI", -2.0, 2.5)],
        wr_scores={"OTC_GDAXI": 0.81},
        best_symbol_only=True,
    )
    assert picked == {"OTC_GDAXI"}
    assert "wr=0.81" in note


def test_select_min_abs_z_skips_when_z_too_weak():
    spreads = {"OTC_SPC": 0.1, "OTC_NDX": 0.2}
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_SPC", "OTC_NDX"],
        TradeDirection.CALL,
        spreads,
        hmm_state=0,
        cfg={"enabled": True, "max_per_cluster": 1, "min_abs_z": 5.0},
    )
    assert picked == set()
    assert note == "STATARB_NO_Z_ALIGN"


def test_symbol_z_supports_put_mean_reversion():
    assert symbol_z_supports_direction(2.5, TradeDirection.PUT, hmm_state=0, min_abs_z=1.2) is True
    assert symbol_z_supports_direction(0.5, TradeDirection.PUT, hmm_state=0, min_abs_z=1.2) is False


def test_symbol_z_supports_trending_put_needs_negative_z():
    assert symbol_z_supports_direction(-1.5, TradeDirection.PUT, hmm_state=1, min_abs_z=1.2) is True
    assert symbol_z_supports_direction(2.5, TradeDirection.PUT, hmm_state=1, min_abs_z=1.2) is False


def test_symbol_z_supports_trending_call_needs_positive_z():
    assert symbol_z_supports_direction(1.5, TradeDirection.CALL, hmm_state=1, min_abs_z=1.2) is True


def test_symbol_z_supports_rejects_invalid_direction_in_trend():
    assert symbol_z_supports_direction(1.5, None, hmm_state=1, min_abs_z=1.2) is False


def test_symbol_z_supports_vetoes_mean_reversion_misalign():
    assert symbol_z_supports_direction(3.0, TradeDirection.CALL, hmm_state=0, z_threshold=2.5, min_abs_z=0.0) is False


def test_select_without_z_align_uses_legacy_filter():
    spreads = {"OTC_SPC": 0.1, "OTC_NDX": 0.2}
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_SPC", "OTC_NDX"],
        TradeDirection.CALL,
        spreads,
        cfg={"enabled": True, "max_per_cluster": 1, "min_abs_z": 0.0},
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
    assert note == "CLUSTER_ALL_SYMBOLS"


def test_resolve_best_symbol_only_forces_single_leader():
    cfg = resolve_statarb_cluster_config(
        {
            "best_symbol_only": True,
            "statarb_index_max_per_cluster": 5,
        },
        None,
    )
    assert cfg["best_symbol_only"] is True
    assert cfg["execute_all"] is False
    assert cfg["enabled"] is True
    assert cfg["max_per_cluster"] == 1


def test_select_prefers_higher_rolling_wr_when_z_similar():
    spreads = {"OTC_FCHI": -2.5, "OTC_GDAXI": -2.4}
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_FCHI", "OTC_GDAXI"],
        TradeDirection.CALL,
        spreads,
        cfg={"enabled": True, "max_per_cluster": 1, "min_abs_z": 0.0, "wr_weight": 0.5},
        wr_scores={"OTC_FCHI": 0.45, "OTC_GDAXI": 0.72},
    )
    assert picked == {"OTC_GDAXI"}
    assert "STATARB_BEST" in note
    assert "wr=0.72" in note


def test_statarb_execute_min_abs_z_uses_config_floor():
    cfg = {"min_abs_z": 0.85}
    assert statarb_execute_min_abs_z("STATARB_BEST leader=OTC_DJI z=-1.10", cfg) == 0.85
    assert statarb_execute_min_abs_z("", cfg) == 0.85


def test_select_cluster_symbol_attempt_order_puts_leader_first():
    spreads = {"OTC_SPC": -2.0, "OTC_NDX": -0.3, "OTC_DJI": 0.5}
    order, note, picked = select_cluster_symbol_attempt_order(
        ["OTC_SPC", "OTC_NDX", "OTC_DJI"],
        TradeDirection.CALL,
        spreads,
        hmm_state=0,
        cfg={"enabled": True, "max_per_cluster": 1, "min_abs_z": 0.0},
    )
    assert picked == {"OTC_SPC"}
    assert order[0] == "OTC_SPC"
    assert "leader=OTC_SPC" in note


def test_resolve_statarb_cluster_config_for_tag():
    cfg = resolve_statarb_cluster_config_for_tag(
        {"statarb_index_min_abs_z": 0.85},
        {"statarb_min_abs_z_by_tag": {"risk_on": 0.50}},
        "risk_on",
    )
    assert cfg["min_abs_z"] == 0.50


def test_select_cluster_symbol_attempt_order_no_candidates():
    order, note, picked = select_cluster_symbol_attempt_order(
        [],
        TradeDirection.CALL,
        {},
        hmm_state=0,
        cfg={"enabled": True},
    )
    assert order == []
    assert picked == set()
    assert "empty" in note.lower()


def test_select_cluster_symbol_attempt_order_disabled_or_execute_all():
    order, note, picked = select_cluster_symbol_attempt_order(
        ["OTC_SPC"],
        TradeDirection.CALL,
        {"OTC_SPC": -2.0},
        hmm_state=0,
        cfg={"enabled": False},
    )
    assert order == ["OTC_SPC"]
    assert note == "CLUSTER_ALL_SYMBOLS"

    order2, note2, picked2 = select_cluster_symbol_attempt_order(
        ["OTC_SPC"],
        TradeDirection.CALL,
        {"OTC_SPC": -2.0},
        hmm_state=0,
        cfg={"enabled": True, "execute_all": True},
    )
    assert order2 == ["OTC_SPC"]
