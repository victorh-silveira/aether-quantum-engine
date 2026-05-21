import json

from src.application.services.llm import llm_trade_parse as ltp
from src.application.services.llm.llm_trade_parse import (
    is_llm_trade_response_complete,
    missing_llm_trade_fields,
    parse_llm_trade_response,
)


def test_parse_format_line_complete():
    raw = "EURUSD: CALL | US_CLUSTER: PUT | EU_CLUSTER: CALL | Probabilidade: 0.85"
    out = parse_llm_trade_response(raw)
    assert out["direction"] == "CALL"
    assert out["us_cluster"] == "PUT"
    assert out["eu_cluster"] == "CALL"
    assert out["conviction"] == 0.85
    assert is_llm_trade_response_complete(out)


def test_parse_json_complete():
    payload = {
        "EURUSD": "PUT",
        "US_CLUSTER": "PUT",
        "EU_CLUSTER": "CALL",
        "Probabilidade": 0.66,
    }
    out = parse_llm_trade_response(json.dumps(payload))
    assert out["direction"] == "PUT"
    assert is_llm_trade_response_complete(out)


def test_parse_incomplete_prose_rejected():
    out = parse_llm_trade_response("MACRO_CONFLUENCIA indica um cenario misto")
    assert not is_llm_trade_response_complete(out)
    assert "EURUSD" in missing_llm_trade_fields(out)


def test_parse_rise_fall_maps_to_call_put():
    raw = "EURUSD: CALL | US_CLUSTER: FALL | EU_CLUSTER: RISE | Probabilidade: 0.65"
    out = parse_llm_trade_response(raw)
    assert out["direction"] == "CALL"
    assert out["us_cluster"] == "PUT"
    assert out["eu_cluster"] == "CALL"
    assert is_llm_trade_response_complete(out)


def test_parse_partial_us_only():
    out = parse_llm_trade_response("EURUSD: CALL | US_CLUSTER: PUT")
    assert out["direction"] == "CALL"
    assert out["us_cluster"] == "PUT"
    assert not is_llm_trade_response_complete(out)
    assert "EU_CLUSTER" in missing_llm_trade_fields(out)


def test_norm_dir_unknown_and_rise_fall_units():
    assert ltp._norm_dir("WAIT") is None
    assert ltp._norm_dir("RISE") == "CALL"
    assert ltp._norm_dir("UP") == "CALL"
    assert ltp._norm_dir("DOWN") == "PUT"


def test_conviction_from_value_invalid_and_percent_scale():
    assert ltp._conviction_from_value("x") == 0.75
    assert ltp._conviction_from_value(None) == 0.75
    assert ltp._conviction_from_value(85) == 0.85


def test_parse_json_empty_and_non_object():
    assert ltp._parse_json_trade("") is None
    assert ltp._parse_json_trade("[1,2]") is None
