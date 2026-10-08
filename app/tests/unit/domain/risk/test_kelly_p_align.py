"""Testes do alinhamento e piso de probabilidade Kelly."""

import pytest

from src.domain.risk.kelly_p_align import (
    apply_kelly_side_p,
    ensure_kelly_edge_p,
    kelly_breakeven_p,
    resolve_kelly_p_floor,
    side_confidence,
)


def test_resolve_kelly_p_floor_clamps():
    assert resolve_kelly_p_floor({"kelly_p_floor": 0.40}) == 0.51
    assert resolve_kelly_p_floor({"kelly_p_floor": 0.90}) == 0.65
    assert resolve_kelly_p_floor({"kelly_p_floor": 0.55}) == 0.55
    assert resolve_kelly_p_floor(None) == 0.55


def test_ensure_kelly_edge_p_respects_breakeven():
    p = ensure_kelly_edge_p(0.55, 0.5, {"kelly_p_floor": 0.55})
    assert p > kelly_breakeven_p(0.5)
    assert (0.5 * p - (1.0 - p)) / 0.5 > 0.0


def test_side_confidence_adapt_transfers_tcn_magnitude():
    metrics = {"scale_adapted": True, "tcn_direction": "PUT"}
    assert side_confidence(0.40, "CALL", metrics=metrics) == 0.60


def test_apply_kelly_side_p_floors_thin_call():
    metrics = {"calibrated_prob": 0.51, "conviction": 0.51}
    p = apply_kelly_side_p(
        metrics,
        order_direction="CALL",
        kelly_config={"kelly_p_floor": 0.55},
        conviction=0.51,
        payout=0.82,
    )
    assert p >= 0.55
    assert metrics["kelly_p_floored"] is True
    assert metrics["kelly_side_p"] == p


def test_apply_kelly_side_p_uses_fusion_p_eff_for_put():
    metrics = {
        "fusion_applied": True,
        "fusion_p_call": 0.55,
        "fusion_p_put": 0.55,
        "fusion_p_eff": 0.90,
        "calibrated_prob": 0.13,
        "exec_direction": "PUT",
    }
    p = apply_kelly_side_p(
        metrics,
        order_direction="PUT",
        kelly_config={"kelly_p_floor": 0.55},
        conviction=0.51,
        payout=0.72,
    )
    assert p == pytest.approx(0.90)
    assert metrics.get("kelly_used_fusion_p_eff") is True
    assert metrics["kelly_side_p"] == pytest.approx(0.90)


def test_apply_kelly_side_p_ignores_invalid_fusion_p_eff():
    metrics = {
        "fusion_applied": True,
        "fusion_p_eff": "bad",
        "calibrated_prob": 0.62,
        "exec_direction": "CALL",
    }
    p = apply_kelly_side_p(
        metrics,
        order_direction="CALL",
        kelly_config={"kelly_p_floor": 0.55},
        conviction=0.51,
        payout=0.95,
    )
    assert metrics.get("kelly_used_fusion_p_eff") is not True
    assert p >= 0.55


def test_kelly_breakeven_non_positive_payout():
    assert kelly_breakeven_p(0.0) == 1.0
    assert kelly_breakeven_p(-1.0) == 1.0
