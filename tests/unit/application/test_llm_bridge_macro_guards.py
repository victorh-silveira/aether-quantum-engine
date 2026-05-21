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
