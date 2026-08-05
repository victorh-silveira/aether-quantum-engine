"""Testes do alinhamento e piso de probabilidade Kelly."""

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


def test_kelly_breakeven_non_positive_payout():
    assert kelly_breakeven_p(0.0) == 1.0
    assert kelly_breakeven_p(-1.0) == 1.0
