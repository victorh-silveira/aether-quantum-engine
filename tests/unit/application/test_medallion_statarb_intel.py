import pytest

from src.application.services.llm.global_macro_confluence import MacroSnapshot
from src.application.services.llm.llm_macro_confluence_guards import apply_macro_confluence_guard
from src.domain.models.trade import TradeDirection


def test_statarb_guard_intelligence_boost_put_and_trending_caution():
    snap_mr = MacroSnapshot(
        tag="divergence_eu_leads",
        eurusd_bias="CALL",
        us_dir="down",
        eu_dir="up",
        us_strength=1.0,
        eu_strength=1.0,
        cluster_status="active",
        macro_block="",
        fx_reference_line="",
        us_parts=("OTC_SPC",),
        eu_parts=("OTC_GDAXI",),
        statarb_spreads={"frxEURUSD": 3.0},
        hmm_state=0,
        hmm_prob=0.9,
    )
    direction, conviction, _, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.80,
        snap_mr,
        {"statarb_z_threshold": 2.5, "macro_intelligence_only": True},
        sym="frxEURUSD",
    )
    assert direction == TradeDirection.PUT
    assert conviction == pytest.approx(0.90)
    assert "STATARB_INTEL boost PUT" in note
    assert execute_ok is True

    snap_tr = MacroSnapshot(
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
        statarb_spreads={"frxEURUSD": 3.0},
        hmm_state=1,
        hmm_prob=0.9,
    )
    _, conv2, _, note2, ok2 = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.80,
        snap_tr,
        {"statarb_z_threshold": 2.5, "macro_intelligence_only": True},
        sym="frxEURUSD",
    )
    assert conv2 == pytest.approx(0.77)
    assert "STATARB_INTEL trending_caution" in note2
    assert ok2 is True


def test_statarb_guard_intelligence_spread_diverge():
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
        statarb_spreads={"frxEURUSD": 3.0},
        hmm_state=0,
        hmm_prob=0.9,
    )
    direction, conviction, _, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.80,
        snap,
        {"statarb_z_threshold": 2.5, "macro_intelligence_only": True},
        sym="frxEURUSD",
    )
    assert direction == TradeDirection.CALL
    assert conviction == pytest.approx(0.76)
    assert "STATARB_INTEL spread_diverge" in note
    assert execute_ok is True


def test_statarb_guard_intelligence_neutral_z_no_adjustment():
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
        statarb_spreads={"frxEURUSD": 0.5},
        hmm_state=0,
        hmm_prob=0.9,
    )
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.80,
        snap,
        {"statarb_z_threshold": 2.5, "macro_intelligence_only": True},
        sym="frxEURUSD",
    )
    assert direction == TradeDirection.CALL
    assert conviction == pytest.approx(0.80)
    assert execute_ok is True
    assert applied is False
    assert note == ""


def test_statarb_guard_intelligence_boost_call_without_block():
    snap = MacroSnapshot(
        tag="divergence_us_leads",
        eurusd_bias="CALL",
        us_dir="up",
        eu_dir="down",
        us_strength=1.0,
        eu_strength=1.0,
        cluster_status="active",
        macro_block="",
        fx_reference_line="",
        us_parts=("OTC_SPC",),
        eu_parts=("OTC_GDAXI",),
        statarb_spreads={"frxEURUSD": -3.0},
        hmm_state=0,
        hmm_prob=0.9,
    )
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.80,
        snap,
        {"statarb_z_threshold": 2.5, "macro_intelligence_only": True},
        sym="frxEURUSD",
    )
    assert direction == TradeDirection.CALL
    assert conviction == pytest.approx(0.90)
    assert execute_ok is True
    assert "STATARB_INTEL boost CALL" in note
