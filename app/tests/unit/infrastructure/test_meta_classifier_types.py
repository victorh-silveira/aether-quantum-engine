import pytest

from src.infrastructure.inference.meta_classifier_types import parse_meta_predict_response


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
