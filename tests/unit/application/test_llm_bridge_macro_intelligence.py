import pytest

from src.application.services.llm.global_macro_confluence import build_macro_snapshot
from src.application.services.llm.llm_macro_confluence_guards import apply_macro_confluence_guard
from src.application.services.llm.symbol_decision_utils import apply_macro_post_parse
from src.domain.models.trade import TradeDirection


def _snapshot(_tag: str, us_dir: str, eu_dir: str) -> object:
    return build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {
            "OTC_SPC": [100.0, 105.0] if us_dir == "up" else [100.0, 95.0],
            "OTC_FCHI": [100.0, 105.0] if eu_dir == "up" else [100.0, 95.0],
        },
        {"min_indices_for_vote": 1, "cluster_return_threshold_pct": 0.02},
    )


def test_apply_macro_guard_intelligence_without_statarb_spread():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 105.0], "OTC_FCHI": [100.0, 95.0]},
        {"min_indices_for_vote": 1},
    )
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.82,
        snap,
        {"macro_intelligence_only": True},
        sym="frxEURUSD",
    )
    assert direction == TradeDirection.CALL
    assert conviction == pytest.approx(0.82)
    assert execute_ok is True
    assert "STATARB" not in note


def test_apply_macro_guard_intelligence_preserves_llm_divergence():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 105.0], "OTC_FCHI": [100.0, 95.0]},
        {"min_indices_for_vote": 1},
    )
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.802,
        snap,
        {"macro_intelligence_only": True},
    )
    assert direction == TradeDirection.CALL
    assert conviction == pytest.approx(0.802)
    assert execute_ok is True
    assert "MACRO_INTEL" in note


def test_apply_macro_guard_intelligence_allows_put_against_us_leader():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 105.0], "OTC_FCHI": [100.0, 95.0]},
        {"min_indices_for_vote": 1},
    )
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.85,
        snap,
        {"macro_intelligence_only": True},
    )
    assert direction == TradeDirection.PUT
    assert conviction == pytest.approx(0.85)
    assert execute_ok is True
    assert "MACRO_DIV_VETO" not in note


def test_apply_macro_post_parse_intelligence_keeps_llm_clusters():
    snap = _snapshot("risk_on", "up", "up")
    direction, conv, note, us_dir, eu_dir, guard, execute_ok = apply_macro_post_parse(
        TradeDirection.CALL,
        0.88,
        "LLM",
        TradeDirection.PUT,
        TradeDirection.CALL,
        snap,
        {"macro_intelligence_only": True, "align_clusters_with_macro_vote": False},
    )
    assert us_dir == TradeDirection.PUT
    assert eu_dir == TradeDirection.CALL
    assert direction == TradeDirection.CALL
    assert conv == pytest.approx(0.88)
    assert execute_ok is True
