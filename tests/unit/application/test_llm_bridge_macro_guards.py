import pytest

from src.application.services.llm.global_macro_confluence import MacroSnapshot, build_macro_snapshot
from src.application.services.llm.llm_bridge_guards import apply_macro_confluence_guard
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


def test_apply_macro_guard_blocks_eurusd_put_against_risk_on():
    snap = _snapshot("risk_on", "up", "up")
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.9,
        snap,
        {
            "align_eurusd_with_confluence": True,
            "confluence_conviction_floor": 0.55,
            "divergence_blocks_execution": True,
            "divergence_max_conviction": 0.65,
        },
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
        {
            "align_eurusd_with_confluence": True,
            "confluence_conviction_floor": 0.55,
        },
    )
    assert applied is False
    assert direction == TradeDirection.PUT
    assert execute_ok is True


def test_apply_macro_guard_divergence_caps_conviction():
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
        {"divergence_blocks_execution": True, "divergence_max_conviction": 0.65},
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


def test_apply_macro_guard_divergence_penalizes_and_caps_below_execute():
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
        {"divergence_blocks_execution": True, "divergence_max_conviction": 0.78},
    )
    assert applied is True
    assert conviction == pytest.approx(0.704)
    assert direction == TradeDirection.CALL
    assert execute_ok is True
    assert "MACRO_DIV cap=0.78" in note


def test_apply_macro_guard_divergence_eu_leads_vetoes_put_against_leader():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 95.0], "OTC_FCHI": [100.0, 105.0]},
        {"min_indices_for_vote": 1},
    )
    assert snap.tag == "divergence_eu_leads"
    direction, _, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.9,
        snap,
        {
            "align_eurusd_with_confluence": True,
            "confluence_conviction_floor": 0.85,
            "divergence_blocks_execution": True,
            "divergence_max_conviction": 0.78,
        },
    )
    assert applied is True
    assert direction is None
    assert execute_ok is False
    assert "MACRO_DIV_VETO" in note


def test_apply_macro_guard_divergence_vetoes_eurusd_against_leader():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 105.0], "OTC_FCHI": [100.0, 95.0]},
        {"min_indices_for_vote": 1},
    )
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.9,
        snap,
        {
            "align_eurusd_with_confluence": True,
            "confluence_conviction_floor": 0.85,
            "divergence_blocks_execution": True,
            "divergence_max_conviction": 0.78,
        },
    )
    assert applied is True
    assert direction is None
    assert execute_ok is False
    assert "MACRO_DIV_VETO" in note
