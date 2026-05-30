from src.application.services.llm.global_macro_confluence import build_macro_snapshot
from src.application.services.llm.llm_refresh_policy import anchor_cached_decision_valid
from src.application.services.llm.macro_cluster_align import (
    align_cluster_dirs_for_divergence_tag,
    quant_trade_direction,
)
from src.application.services.llm.symbol_decision_utils import apply_macro_post_parse
from src.domain.models.trade import TradeDirection
from tests.unit.application.macro_guard_fixtures import RELAXED_MACRO_CFG


def test_align_us_leader_fallback_when_llm_missing():
    us_dir, eu_dir = align_cluster_dirs_for_divergence_tag(
        "divergence_us_leads",
        us_dir_quant="up",
        eu_dir_quant="down",
        us_dir=None,
        eu_dir=TradeDirection.PUT,
    )
    assert us_dir == TradeDirection.CALL
    assert eu_dir == TradeDirection.PUT


def test_align_keeps_llm_when_present():
    us_dir, eu_dir = align_cluster_dirs_for_divergence_tag(
        "divergence_us_leads",
        us_dir_quant="up",
        eu_dir_quant="down",
        us_dir=TradeDirection.PUT,
        eu_dir=TradeDirection.PUT,
    )
    assert us_dir == TradeDirection.PUT
    assert eu_dir == TradeDirection.PUT


def test_align_eu_leader_fallback_when_llm_missing():
    us_dir, eu_dir = align_cluster_dirs_for_divergence_tag(
        "divergence_eu_leads",
        us_dir_quant="up",
        eu_dir_quant="down",
        us_dir=TradeDirection.CALL,
        eu_dir=None,
    )
    assert us_dir == TradeDirection.CALL
    assert eu_dir == TradeDirection.PUT


def test_align_divergence_eu_keeps_llm_eu_direction():
    us_dir, eu_dir = align_cluster_dirs_for_divergence_tag(
        "divergence_eu_leads",
        us_dir_quant="up",
        eu_dir_quant="down",
        us_dir=TradeDirection.CALL,
        eu_dir=TradeDirection.PUT,
    )
    assert us_dir == TradeDirection.CALL
    assert eu_dir == TradeDirection.PUT


def test_quant_trade_direction_flat_returns_none():
    assert quant_trade_direction("flat") is None


def test_apply_macro_post_parse_keeps_llm_cluster_on_divergence():
    snap = build_macro_snapshot(
        ["R_50"],
        ["R_75"],
        {"R_50": [100.0, 105.0], "R_75": [100.0, 95.0]},
        {"min_indices_for_vote": 1, "cluster_return_threshold_pct": 0.02},
    )
    assert snap.tag == "divergence_us_leads"
    _, _, _, us_dir, eu_dir, _, execute_ok = apply_macro_post_parse(
        TradeDirection.CALL,
        0.75,
        "LLM",
        TradeDirection.PUT,
        TradeDirection.PUT,
        snap,
        {**RELAXED_MACRO_CFG, "allowed_execute_tags": ("divergence_us_leads",)},
    )
    assert us_dir == TradeDirection.PUT
    assert execute_ok is True


def test_anchor_cached_decision_valid():
    assert anchor_cached_decision_valid(None, "R_100") is False
    assert anchor_cached_decision_valid({"R_100": {"direction": None}}, "R_100") is False
    assert anchor_cached_decision_valid({"R_100": "bad"}, "R_100") is False
    assert (
        anchor_cached_decision_valid(
            {"R_100": {"direction": TradeDirection.CALL}},
            "R_100",
        )
        is True
    )
