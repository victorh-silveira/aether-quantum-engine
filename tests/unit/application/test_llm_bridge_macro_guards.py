import pytest

from src.application.services.llm.global_macro_confluence import MacroSnapshot, build_macro_snapshot
from src.application.services.llm.llm_macro_confluence_guards import (
    apply_macro_confluence_guard,
    divergence_leader_strength,
)
from src.application.services.llm.symbol_decision_utils import apply_macro_post_parse
from src.domain.models.trade import TradeDirection


def _snapshot(tag: str, us_dir: str, eu_dir: str, us_s: float = 1.0, eu_s: float = 1.0) -> MacroSnapshot:
    return build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {
            "OTC_SPC": [100.0, 105.0] if us_dir == "up" else [100.0, 95.0],
            "OTC_FCHI": [100.0, 105.0] if eu_dir == "up" else [100.0, 95.0],
        },
        {"min_indices_for_vote": 1, "cluster_return_threshold_pct": 0.02},
    )


def _legacy_cfg(**overrides) -> dict:
    base = {"macro_intelligence_only": False, "align_eurusd_with_confluence": True}
    base.update(overrides)
    return base


def test_apply_macro_guard_blocks_eurusd_put_against_risk_on():
    snap = _snapshot("risk_on", "up", "up")
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.9,
        snap,
        _legacy_cfg(
            confluence_conviction_floor=0.55,
            divergence_blocks_execution=True,
            divergence_max_conviction=0.65,
        ),
    )
    assert applied is True
    assert direction is None
    assert execute_ok is False
    assert "MACRO_ALIGN" in note


def test_apply_macro_guard_allows_eurusd_put_on_risk_off():
    snap = _snapshot("risk_off", "down", "down")
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.9,
        snap,
        _legacy_cfg(confluence_conviction_floor=0.55),
    )
    assert applied is False
    assert direction == TradeDirection.PUT
    assert execute_ok is True


def test_apply_macro_guard_divergence_caps_conviction_legacy():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 105.0], "OTC_FCHI": [100.0, 95.0]},
        {"min_indices_for_vote": 1},
    )
    assert snap.tag == "divergence_us_leads"
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.9,
        snap,
        _legacy_cfg(divergence_blocks_execution=True, divergence_max_conviction=0.65),
    )
    assert applied is True
    assert conviction == pytest.approx(0.65)
    assert direction == TradeDirection.CALL
    assert execute_ok is True
    assert "MACRO_DIV" in note


def test_apply_macro_post_parse_aligns_us_cluster_on_risk_on():
    snap = _snapshot("risk_on", "up", "up")
    direction, conv, note, us_dir, eu_dir, guard, execute_ok = apply_macro_post_parse(
        TradeDirection.CALL,
        0.65,
        "LLM",
        TradeDirection.PUT,
        TradeDirection.CALL,
        snap,
        {"align_clusters_with_macro_vote": True, "confluence_conviction_floor": 0.55},
    )
    assert us_dir == TradeDirection.CALL
    assert eu_dir == TradeDirection.CALL
    assert guard is True
    assert "MACRO_CLUSTER_ALIGN" in note
    assert execute_ok is True


def test_apply_macro_post_parse_clears_us_cluster_when_quant_flat():
    snap = build_macro_snapshot(
        ["OTC_SPC", "OTC_NDX", "OTC_DJI"],
        ["OTC_FCHI", "OTC_GDAXI", "OTC_FTSE"],
        {
            "OTC_SPC": [100.0, 100.05],
            "OTC_NDX": [100.0, 100.04],
            "OTC_DJI": [100.0, 100.03],
            "OTC_FCHI": [100.0, 95.0],
            "OTC_GDAXI": [100.0, 94.5],
            "OTC_FTSE": [100.0, 94.0],
        },
        {"min_indices_for_vote": 2, "cluster_min_move_pct": 0.10, "cluster_return_threshold_pct": 0.02},
    )
    assert snap.tag == "indefinido"
    assert snap.us_dir == "flat"
    assert snap.eu_dir == "down"
    direction, conv, note, us_dir, eu_dir, guard, _ = apply_macro_post_parse(
        TradeDirection.CALL,
        0.68,
        "LLM",
        TradeDirection.CALL,
        TradeDirection.PUT,
        snap,
        {"align_clusters_with_macro_vote": True, "confluence_conviction_floor": 0.55},
    )
    assert us_dir is None
    assert eu_dir == TradeDirection.PUT
    assert guard is True
    assert "MACRO_US_SKIP" in note or "MACRO_CLUSTER_ALIGN" in note


def test_apply_macro_post_parse_indefinido_forces_us_put_when_quant_down():
    snap = build_macro_snapshot(
        ["OTC_SPC", "OTC_NDX", "OTC_DJI"],
        ["OTC_FCHI"],
        {
            "OTC_SPC": [100.0, 95.0],
            "OTC_NDX": [100.0, 94.0],
            "OTC_DJI": [100.0, 93.5],
            "OTC_FCHI": [100.0, 100.05],
        },
        {"min_indices_for_vote": 2, "cluster_min_move_pct": 0.10},
    )
    snap = type(snap)(
        us_dir="down",
        eu_dir="flat",
        us_strength=snap.us_strength,
        eu_strength=0.0,
        tag="indefinido",
        eurusd_bias=snap.eurusd_bias,
        cluster_status=snap.cluster_status,
        macro_block=snap.macro_block,
        fx_reference_line=snap.fx_reference_line,
        us_parts=snap.us_parts,
        eu_parts=snap.eu_parts,
    )
    _, _, note, us_dir, eu_dir, _, _ = apply_macro_post_parse(
        TradeDirection.CALL,
        0.68,
        "LLM",
        TradeDirection.CALL,
        TradeDirection.PUT,
        snap,
        {"align_clusters_with_macro_vote": True, "confluence_conviction_floor": 0.55},
    )
    assert us_dir == TradeDirection.PUT
    assert eu_dir is None


def test_apply_macro_guard_legacy_divergence_eu_leads_blends_mcs():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 95.0], "OTC_FCHI": [100.0, 105.0]},
        {"min_indices_for_vote": 1},
    )
    assert snap.tag == "divergence_eu_leads"
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.9,
        snap,
        _legacy_cfg(divergence_blocks_execution=False),
    )
    assert direction == TradeDirection.CALL
    assert conviction > 0.75
    assert execute_ok is True
    assert applied is False or note == ""


def test_divergence_leader_strength_unknown_tag_returns_zero():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 105.0], "OTC_FCHI": [100.0, 105.0]},
        {"min_indices_for_vote": 1},
    )
    assert divergence_leader_strength(snap, "risk_on") == 0.0


def test_apply_macro_guard_legacy_no_veto_when_align_disabled():
    snap = _snapshot("risk_on", "up", "up")
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.9,
        snap,
        _legacy_cfg(align_eurusd_with_confluence=False, divergence_blocks_execution=False),
    )
    assert direction == TradeDirection.PUT
    assert execute_ok is True
    assert "MACRO_ALIGN" not in note


def test_apply_statarb_no_spread_and_neutral_z_legacy():
    snap = MacroSnapshot(
        tag="risk_on",
        eurusd_bias="CALL",
        us_dir="up",
        eu_dir="up",
        us_strength=1.0,
        eu_strength=1.0,
        cluster_status="active",
        macro_block="",
        fx_reference_line="",
        us_parts=("OTC_SPC",),
        eu_parts=("OTC_GDAXI",),
        statarb_spreads={"OTC_GDAXI": 0.5},
        hmm_state=0,
        hmm_prob=0.9,
    )
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.60,
        snap,
        {"macro_intelligence_only": False, "statarb_z_threshold": 2.5},
        sym="OTC_GDAXI",
    )
    assert direction == TradeDirection.CALL
    assert execute_ok is True
    assert applied is False or note == ""


def test_apply_macro_guard_divergence_vetoes_eurusd_against_leader_legacy():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 105.0], "OTC_FCHI": [100.0, 95.0]},
        {"min_indices_for_vote": 1},
    )
    direction, _, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.9,
        snap,
        _legacy_cfg(
            confluence_conviction_floor=0.85,
            divergence_blocks_execution=True,
            divergence_max_conviction=0.78,
        ),
    )
    assert applied is True
    assert direction is None
    assert execute_ok is False
    assert "MACRO_DIV_VETO" in note
