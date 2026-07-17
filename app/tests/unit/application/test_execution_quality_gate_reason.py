from src.application.services.execution_quality_gate_reason import (
    build_quality_gate_reason,
    edge_reject_clause,
    margin_reject_clause,
)


def test_quality_gate_reason_builds_margin_only():
    reason = build_quality_gate_reason(
        dir_margin=0.05,
        min_margin=0.08,
        payoff_edge=0.01,
        min_edge=0.04,
        margin_fail=True,
        edge_fail=True,
        meta_applied=True,
    )
    assert reason == "[TCN Margin 0.05 < min 0.08]"


def test_quality_gate_reason_ignores_edge_when_meta_not_applied():
    reason = build_quality_gate_reason(
        dir_margin=0.05,
        min_margin=0.08,
        payoff_edge=0.01,
        min_edge=0.04,
        margin_fail=True,
        edge_fail=True,
        meta_applied=False,
    )
    assert reason == "[TCN Margin 0.05 < min 0.08]"


def test_quality_gate_reject_clause_helpers():
    assert margin_reject_clause(0.07, 0.10) == "[TCN Margin 0.07 < min 0.10]"
    assert edge_reject_clause(0.02, 0.04) == "[Meta Payoff 0.02 < min 0.04]"


def test_quality_gate_reason_prefers_neutral_clamp():
    reason = build_quality_gate_reason(
        dir_margin=0.00,
        min_margin=0.15,
        payoff_edge=0.01,
        min_edge=0.70,
        margin_fail=True,
        edge_fail=True,
        meta_applied=True,
        calibration_mode="neutral_clamp",
    )
    assert reason == "neutral_clamp"


def test_quality_gate_reason_empty_when_no_margin_fail():
    reason = build_quality_gate_reason(
        dir_margin=0.20,
        min_margin=0.15,
        payoff_edge=-2.0,
        min_edge=0.70,
        margin_fail=False,
        edge_fail=True,
        meta_applied=True,
    )
    assert reason == ""
