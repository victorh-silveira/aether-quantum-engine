"""Testes dos filtros assertivos Medallion."""

import pytest

from src.application.services.llm import llm_macro_confluence_guards as macro_guards
from src.application.services.llm.llm_macro_confluence_guards import (
    _cluster_floor_ok,
    apply_macro_confluence_guard,
)
from src.application.services.llm.macro_config import MacroSnapshot
from src.domain.models.trade import TradeDirection
from src.domain.risk.risk_manager import RiskManager


def _snap(tag: str, *, us_s: float = 0.9, eu_s: float = 0.88) -> MacroSnapshot:
    return MacroSnapshot(
        us_dir="up",
        eu_dir="up",
        us_strength=us_s,
        eu_strength=eu_s,
        tag=tag,
        eurusd_bias="CALL",
        cluster_status="",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
        statarb_spreads={"OTC_SPC": -2.8},
        hmm_state=0,
        hmm_prob=0.9,
    )


def test_divergence_veto_weak_leader():
    snap = _snap("divergence_us_leads", us_s=0.55, eu_s=0.40)
    _, _, _, note, ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.75,
        snap,
        {"divergence_min_leader_strength": 0.70, "confluence_conviction_floor": 0.65},
        sym="OTC_SPC",
    )
    assert ok is False
    assert "divergence_leader" in note


def test_divergence_veto_us_gap():
    snap = _snap("divergence_us_leads", us_s=0.70, eu_s=0.68)
    _, _, _, note, ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.75,
        snap,
        {
            "divergence_min_leader_strength": 0.60,
            "divergence_min_strength_gap": 0.05,
            "confluence_conviction_floor": 0.0,
        },
    )
    assert ok is False
    assert "divergence_us_gap" in note


def test_divergence_veto_eu_gap():
    snap = _snap("divergence_eu_leads", us_s=0.68, eu_s=0.70)
    _, _, _, note, ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.75,
        snap,
        {
            "divergence_min_leader_strength": 0.60,
            "divergence_min_strength_gap": 0.05,
            "confluence_conviction_floor": 0.0,
        },
    )
    assert ok is False
    assert "divergence_eu_gap" in note


def test_divergence_cap_note_still_executes():
    snap = _snap("divergence_us_leads", us_s=0.90, eu_s=0.50)
    _, conv, _, note, ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.95,
        snap,
        {
            "divergence_min_leader_strength": 0.0,
            "divergence_min_strength_gap": 0.0,
            "divergence_max_conviction": 0.80,
            "confluence_conviction_floor": 0.0,
        },
    )
    assert ok is True
    assert "MACRO_CAP" in note
    assert conv == pytest.approx(0.80)


def test_macro_hmm_prob_veto():
    snap = _snap("risk_on")
    snap = MacroSnapshot(
        us_dir=snap.us_dir,
        eu_dir=snap.eu_dir,
        us_strength=snap.us_strength,
        eu_strength=snap.eu_strength,
        tag=snap.tag,
        eurusd_bias=snap.eurusd_bias,
        cluster_status=snap.cluster_status,
        macro_block=snap.macro_block,
        fx_reference_line=snap.fx_reference_line,
        us_parts=snap.us_parts,
        eu_parts=snap.eu_parts,
        hmm_prob=0.40,
    )
    _, _, _, note, ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.80,
        snap,
        {"assert_min_hmm_prob": 0.55, "confluence_conviction_floor": 0.0},
    )
    assert ok is False
    assert "hmm_prob" in note


def test_risk_on_veto_when_tag_not_allowed():
    snap = _snap("risk_on")
    _, _, _, note, ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.80,
        snap,
        {
            "confluence_conviction_floor": 0.0,
            "allowed_execute_tags": ("risk_off", "divergence_us_leads", "divergence_eu_leads"),
        },
        sym="OTC_NDX",
    )
    assert ok is False
    assert "tag_not_allowed" in note


def test_risk_on_and_off_weak_clusters():
    snap = _snap("risk_on", us_s=0.40, eu_s=0.90)
    _, _, _, note_on, ok_on = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.80,
        snap,
        {"confluence_conviction_floor": 0.55},
    )
    assert ok_on is False
    assert "risk_on_weak_clusters" in note_on

    snap_off = _snap("risk_off", us_s=0.40, eu_s=0.42)
    _, _, _, note_off, ok_off = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.80,
        snap_off,
        {"confluence_conviction_floor": 0.55},
    )
    assert ok_off is False
    assert "risk_off_weak_clusters" in note_off


def test_cluster_floor_ok_unknown_tag_returns_false():
    snap = _snap("custom_tag", us_s=1.0, eu_s=1.0)
    assert _cluster_floor_ok(snap, 0.5) is False


def test_risk_on_allows_when_us_cluster_strong_even_if_eu_weak():
    snap = _snap("risk_on", us_s=0.72, eu_s=0.30)
    _, _, _, _, ok_on = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.80,
        snap,
        {"confluence_conviction_floor": 0.65},
    )
    assert ok_on is True


def test_risk_off_allows_when_eu_cluster_strong_even_if_us_weak():
    snap = _snap("risk_off", us_s=0.30, eu_s=0.72)
    _, _, _, _, ok_off = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.80,
        snap,
        {"confluence_conviction_floor": 0.65},
    )
    assert ok_off is True


def test_macro_unknown_tag_veto():
    snap = _snap("custom_tag")
    _, _, _, note, ok = apply_macro_confluence_guard(TradeDirection.CALL, 0.80, snap, {})
    assert ok is False
    assert "unknown_tag" in note


def test_statarb_spread_diverge_without_direction():
    direction, conv, applied, notes, ok = macro_guards._apply_statarb_intelligence(
        None,
        0.80,
        3.0,
        2.5,
        0,
    )
    assert direction is None
    assert conv == pytest.approx(0.74)
    assert applied is True
    assert ok is False
    assert any("spread_diverge" in n for n in notes)


def test_stake_not_blocked_after_large_session_loss():
    rm = RiskManager(
        {
            "kelly": {"fraction": 0.22, "max_stake_pct": 0.09},
            "params": {"payout_estimate": 1.0, "stake_min": 1.0},
        }
    )
    rm.set_initial_bankroll(50.0)
    rm.total_session_profit = -12.0
    stake = rm.calculate_stake(38.0, "OTC_FCHI", 0.7, silent=True)
    assert stake > 0.0
