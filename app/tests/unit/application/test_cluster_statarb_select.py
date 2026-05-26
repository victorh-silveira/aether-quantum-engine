from src.application.services.llm.cluster_statarb_select import (
    _alignment_score,
    _statarb_leader_pick,
    _wr_blend_score,
    resolve_statarb_cluster_config,
    select_cluster_symbols_by_statarb,
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
    assert _alignment_score(-2.0, TradeDirection.CALL, 1) == 1.0
    assert _alignment_score(2.0, TradeDirection.PUT, 1) == 1.0
    assert _alignment_score(1.0, None, 0) == 0.0


def test_soft_fallback_skips_misaligned_and_keeps_valid_row():
    spreads = {"OTC_FCHI": 3.5, "OTC_GDAXI": 0.85}
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_FCHI", "OTC_GDAXI"],
        TradeDirection.CALL,
        spreads,
        hmm_state=0,
        cfg={
            "enabled": True,
            "max_per_cluster": 1,
            "min_abs_z": 1.2,
            "require_z_align": True,
            "z_align_soft_fallback": True,
        },
    )
    assert picked == {"OTC_GDAXI"}
    assert "STATARB_SOFT" in note


def test_soft_fallback_returns_no_align_when_only_misaligned_rows():
    spreads = {"OTC_FCHI": 3.5}
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_FCHI"],
        TradeDirection.CALL,
        spreads,
        hmm_state=0,
        cfg={
            "enabled": True,
            "max_per_cluster": 1,
            "min_abs_z": 1.2,
            "require_z_align": True,
            "z_align_soft_fallback": True,
        },
    )
    assert picked == set()
    assert note == "STATARB_NO_Z_ALIGN"


def test_statarb_leader_pick_includes_wr_in_note():
    picked, note = _statarb_leader_pick(
        [("OTC_GDAXI", -2.0, 2.5)],
        wr_scores={"OTC_GDAXI": 0.81},
        best_symbol_only=True,
    )
    assert picked == {"OTC_GDAXI"}
    assert "wr=0.81" in note


def test_select_soft_fallback_when_strict_align_empty():
    spreads = {"OTC_FCHI": 0.95, "OTC_GDAXI": 0.70}
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_FCHI", "OTC_GDAXI"],
        TradeDirection.PUT,
        spreads,
        hmm_state=0,
        cfg={
            "enabled": True,
            "max_per_cluster": 1,
            "min_abs_z": 1.2,
            "require_z_align": True,
            "z_align_soft_fallback": True,
        },
    )
    assert picked == {"OTC_FCHI"}
    assert "STATARB_SOFT" in note


def test_select_min_abs_z_skips_when_z_too_weak():
    spreads = {"OTC_SPC": 0.1, "OTC_NDX": 0.2}
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_SPC", "OTC_NDX"],
        TradeDirection.CALL,
        spreads,
        hmm_state=0,
        cfg={"enabled": True, "max_per_cluster": 1, "min_abs_z": 5.0, "require_z_align": True},
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
        cfg={"enabled": True, "max_per_cluster": 1, "min_abs_z": 0.0, "require_z_align": False},
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


def test_resolve_execute_all_disables_statarb_index_select():
    cfg = resolve_statarb_cluster_config(
        {"execute_all_cluster_symbols": True, "statarb_index_select_enabled": True},
        None,
    )
    assert cfg["execute_all"] is True
    assert cfg["enabled"] is False


def test_resolve_best_symbol_only_forces_single_leader():
    cfg = resolve_statarb_cluster_config(
        {
            "best_symbol_only": True,
            "execute_all_cluster_symbols": True,
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
        cfg={"enabled": True, "max_per_cluster": 1, "min_abs_z": 0.0, "wr_weight": 0.5, "best_symbol_only": True},
        wr_scores={"OTC_FCHI": 0.45, "OTC_GDAXI": 0.72},
    )
    assert picked == {"OTC_GDAXI"}
    assert "STATARB_BEST" in note
    assert "wr=0.72" in note


def test_execute_all_cluster_symbols_returns_every_candidate():
    picked, note = select_cluster_symbols_by_statarb(
        ["OTC_FCHI", "OTC_GDAXI", "OTC_SSMI"],
        TradeDirection.PUT,
        {"OTC_FCHI": 0.1},
        cfg={"execute_all": True, "enabled": True, "max_per_cluster": 1},
    )
    assert picked == {"OTC_FCHI", "OTC_GDAXI", "OTC_SSMI"}
    assert note == "CLUSTER_ALL_SYMBOLS"
