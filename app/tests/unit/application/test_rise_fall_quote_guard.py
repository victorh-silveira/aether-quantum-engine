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
