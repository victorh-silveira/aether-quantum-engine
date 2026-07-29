import pytest

from src.infrastructure.inference.meta_classifier_types import (
    parse_meta_predict_response,
    resolve_predicted_edge,
)


def test_parse_meta_predict_response_extracts_edge_and_applied():
    parsed = parse_meta_predict_response({"predicted_payoff_edge": 0.14, "meta_applied": True})
    assert parsed["predicted_payoff_edge"] == pytest.approx(0.14)
    assert parsed["meta_applied"] is True
    assert parsed["edge_expectancy"] == "WIN_EXPECTED"


def test_parse_meta_predict_response_defaults_meta_applied_false():
    parsed = parse_meta_predict_response({"predicted_payoff_edge": -0.05})
    assert parsed["meta_applied"] is False
    assert parsed["edge_expectancy"] == "LOSS_EXPECTED"


def test_parse_meta_predict_response_reads_explicit_expectancy():
    parsed = parse_meta_predict_response(
        {"predicted_payoff_edge": 1.12, "meta_applied": True, "edge_expectancy": "NO_EDGE_NEUTRAL"},
    )
    assert parsed["edge_expectancy"] == "NO_EDGE_NEUTRAL"


def test_parse_meta_predict_response_derives_neutral_expectancy_from_low_edge():
    parsed = parse_meta_predict_response({"predicted_payoff_edge": 0.02})
    assert parsed["edge_expectancy"] == "NO_EDGE_NEUTRAL"


def test_parse_meta_predict_response_rejects_missing_edge():
    with pytest.raises(KeyError):
        parse_meta_predict_response({"meta_applied": True})


def test_parse_meta_predict_response_rejects_non_object():
    with pytest.raises(TypeError):
        parse_meta_predict_response([])


class TestResolvePredictedEdge:
    def test_returns_edge_from_calibrated_prob(self):
        edge = resolve_predicted_edge({"calibrated_prob": 0.7})
        assert edge == pytest.approx((0.7 * 1.95) - 1.0)

    def test_falls_back_to_raw_prob(self):
        edge = resolve_predicted_edge({"raw_prob": 0.65})
        assert edge == pytest.approx((0.65 * 1.95) - 1.0)

    def test_defaults_to_0_5_when_no_prob_key(self):
        assert resolve_predicted_edge({}) == pytest.approx((0.5 * 1.95) - 1.0)

    def test_returns_0_for_non_dict_input(self):
        assert resolve_predicted_edge(None) == 0.0  # type: ignore[arg-type]
        assert resolve_predicted_edge("invalid") == 0.0  # type: ignore[arg-type]

    def test_returns_0_when_prob_is_none(self):
        assert resolve_predicted_edge({"calibrated_prob": None, "raw_prob": None}) == 0.0

    def test_uses_dominant_prob_when_below_0_5(self):
        edge = resolve_predicted_edge({"raw_prob": 0.3})
        assert edge == pytest.approx((0.7 * 1.95) - 1.0)

    def test_accepts_custom_payout(self):
        edge = resolve_predicted_edge({"calibrated_prob": 0.8}, payout=0.90)
        assert edge == pytest.approx((0.8 * 1.90) - 1.0)
