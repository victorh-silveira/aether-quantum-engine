import pytest

from src.application.services.llm.global_macro_confluence import MacroSnapshot, build_macro_snapshot
from src.application.services.llm.llm_macro_confluence_guards import (
    apply_macro_confluence_guard,
    divergence_leader_strength,
)
from src.domain.models.trade import TradeDirection


def _snapshot(tag: str, us_dir: str, eu_dir: str) -> MacroSnapshot:
    return build_macro_snapshot(
        ["R_25"],
        ["R_75"],
        {
            "R_25": [100.0, 105.0] if us_dir == "up" else [100.0, 95.0],
            "R_75": [100.0, 105.0] if eu_dir == "up" else [100.0, 95.0],
        },
        {"min_indices_for_vote": 1, "cluster_return_threshold_pct": 0.02},
    )


def test_apply_macro_guard_preserves_llm_direction():
    snap = _snapshot("risk_on", "up", "up")
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.PUT,
        0.9,
        snap,
        {"statarb_z_threshold": 2.5},
    )
    assert direction == TradeDirection.PUT
    assert execute_ok is True
    assert conviction == pytest.approx(0.9)


def test_apply_macro_guard_statarb_boost_call():
    snap = build_macro_snapshot(
        ["R_25"],
        ["R_75"],
        {"R_25": [100.0, 95.0], "R_75": [100.0, 95.0]},
        {"min_indices_for_vote": 1},
        statarb_spreads={"R_25": -3.0},
        hmm_state=0,
    )
    direction, conviction, applied, note, execute_ok = apply_macro_confluence_guard(
        TradeDirection.CALL,
        0.7,
        snap,
        {"statarb_z_threshold": 2.5},
        sym="R_25",
    )
    assert direction == TradeDirection.CALL
    assert execute_ok is True
    assert applied is True
    assert conviction > 0.7
    assert "STATARB_INTEL boost CALL" in note


def test_divergence_leader_strength():
    snap = _snapshot("divergence_us_leads", "up", "flat")
    assert divergence_leader_strength(snap, "divergence_us_leads") >= 0.0
    assert divergence_leader_strength(snap, "divergence_eu_leads") == float(snap.eu_strength)
    assert divergence_leader_strength(snap, "risk_on") == 0.0
