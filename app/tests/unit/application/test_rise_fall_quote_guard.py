"""Edge da cotacao do contrato, sem extrapolar probabilidade de CALL/PUT."""

import pytest

from src.application.services.rise_fall_quote_guard import quoted_edge


@pytest.mark.parametrize(
    ("metrics", "direction", "rate", "expected"),
    [
        ({"calibrated_prob": 0.60}, "CALL", 0.8, 0.08),
        ({"calibrated_prob": 0.40}, "PUT", 0.8, 0.08),
        ({"calibrated_prob": 0.49}, "CALL", 0.8, -0.118),
        ({"calibrated_prob": 0.60}, "PUT", 0.8, -0.28),
        ({}, "CALL", 0.8, None),
        (None, "CALL", 0.8, None),
        ({"calibrated_prob": 0.6}, "BAD", 0.8, None),
        ({"calibrated_prob": "bad"}, "CALL", 0.8, None),
        ({"calibrated_prob": 0.6}, "CALL", None, None),
        ({"calibrated_prob": 0.6}, "CALL", float("nan"), None),
        ({"calibrated_prob": 1.1}, "CALL", 0.8, None),
        ({"calibrated_prob": 0.6}, "CALL", 0.0, None),
        ({"calibrated_prob": 0.40, "loss_clf_flip": True, "loss_clf_p_eff": 0.65}, "CALL", 0.8, 0.17),
        ({"calibrated_prob": 0.40, "loss_clf_flip": True, "loss_clf_p_loss": 0.65}, "CALL", 0.8, 0.17),
        ({"calibrated_prob": 0.40, "loss_clf_flip": True, "loss_clf_p_eff": "bad"}, "CALL", 0.8, -0.28),
        ({"calibrated_prob": 0.40, "loss_clf_flip": True, "loss_clf_p_eff": 1.5}, "CALL", 0.8, None),
        ({"calibrated_prob": 0.52, "anti_trend_lock_flip": True, "conviction": 0.58}, "PUT", 0.8, -0.136),
        ({"calibrated_prob": 0.52, "anti_trend_lock_flip": True, "conviction": "bad"}, "PUT", 0.8, -0.136),
    ],
)
def test_quoted_edge(metrics, direction, rate, expected):
    result = quoted_edge(metrics, direction, rate)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


@pytest.mark.parametrize("haircut", [float("nan"), -0.01, 0.5])
def test_quoted_edge_rejects_invalid_haircut(haircut):
    assert quoted_edge({"calibrated_prob": 0.62}, "CALL", 0.8, probability_haircut=haircut) is None


def test_probability_haircut_reduces_call_and_put_edge_symmetrically():
    for p_call, direction in ((0.62, "CALL"), (0.38, "PUT")):
        edge = quoted_edge({"calibrated_prob": p_call}, direction, 0.8, probability_haircut=0.02)
        assert edge == pytest.approx(0.08)


def test_calculate_payout_breakeven_prob():
    from src.application.services.rise_fall_quote_guard import calculate_payout_breakeven_prob

    assert calculate_payout_breakeven_prob(None) is None
    assert calculate_payout_breakeven_prob("not_a_number") is None
    assert calculate_payout_breakeven_prob(-0.5) is None
    assert calculate_payout_breakeven_prob(0.0) is None
    assert calculate_payout_breakeven_prob(1.0) == pytest.approx(0.50)
    assert calculate_payout_breakeven_prob(0.80) == pytest.approx(1.0 / 1.80)


def test_is_quote_edge_acceptable():
    from src.application.services.rise_fall_quote_guard import is_quote_edge_acceptable

    ok, ev, reason = is_quote_edge_acceptable({}, "CALL", 0.80)
    assert ok is False
    assert ev is None
    assert reason == "insufficient_evidence"

    metrics_low = {"calibrated_prob": 0.50}
    ok, ev, reason = is_quote_edge_acceptable(metrics_low, "CALL", 0.80, min_edge=0.05)
    assert ok is False
    assert reason == "quote_edge_below_min"

    metrics_be = {"calibrated_prob": 0.56}
    ok, ev, reason = is_quote_edge_acceptable(metrics_be, "CALL", 0.80, min_edge=-0.10, safety_margin=0.03)
    assert ok is False
    assert reason == "below_payout_breakeven_margin"

    metrics_ok = {"calibrated_prob": 0.62}
    ok, ev, reason = is_quote_edge_acceptable(metrics_ok, "CALL", 0.80, min_edge=0.02, safety_margin=0.02)
    assert ok is True
    assert ev is not None and ev > 0.05
    assert reason == "ok"
