from src.application.services.llm.global_macro_confluence import (
    MacroSnapshot,
    aggregate_cluster_vote,
    build_macro_snapshot,
    classify_transatlantic_confluence,
    cluster_direction_from_closes,
    cluster_trade_direction,
    empty_macro_snapshot,
    eurusd_bias_from_confluence,
    format_macro_confluence_block,
    fx_reference_context_line,
    resolve_macro_config,
)


def test_cluster_direction_from_closes_thresholds():
    assert cluster_direction_from_closes([100.0, 103.0], 0.02) == "up"
    assert cluster_direction_from_closes([100.0, 97.0], 0.02) == "down"
    assert cluster_direction_from_closes([100.0, 100.01], 0.02) == "flat"
    assert cluster_direction_from_closes([100.0], 0.02) == "flat"
    assert cluster_direction_from_closes([0.0, 1.0], 0.02) == "flat"


def test_classify_transatlantic_confluence_tags():
    assert classify_transatlantic_confluence("up", "up") == "risk_on"
    assert classify_transatlantic_confluence("down", "down") == "risk_off"
    assert classify_transatlantic_confluence("up", "down") == "divergence_us_leads"
    assert classify_transatlantic_confluence("down", "up") == "divergence_eu_leads"
    assert classify_transatlantic_confluence("flat", "up") == "indefinido"
    assert classify_transatlantic_confluence("unknown", "up") == "indefinido"


def test_eurusd_bias_and_cluster_trade_direction():
    assert eurusd_bias_from_confluence("risk_on") == "CALL"
    assert eurusd_bias_from_confluence("risk_off") == "PUT"
    assert eurusd_bias_from_confluence("divergence_us_leads") == "CALL"
    assert eurusd_bias_from_confluence("divergence_eu_leads") == "CALL"
    assert eurusd_bias_from_confluence("indefinido", us_dir="down", eu_dir="flat") == "PUT"
    assert eurusd_bias_from_confluence("indefinido", us_dir="up", eu_dir="flat") == "CALL"
    assert cluster_trade_direction("up") == "CALL"
    assert cluster_trade_direction("down") == "PUT"
    assert cluster_trade_direction("flat") is None


def test_aggregate_cluster_vote_majority():
    symbols = ["OTC_SPC", "OTC_NDX", "OTC_DJI"]
    closes_map = {
        "OTC_SPC": [100.0, 105.0],
        "OTC_NDX": [100.0, 104.0],
        "OTC_DJI": [100.0, 103.0],
    }
    vote = aggregate_cluster_vote(symbols, closes_map, threshold_pct=0.02, min_indices=2)
    assert vote.direction == "up"
    assert vote.strength >= 0.66


def test_build_macro_snapshot_risk_off():
    us = ["OTC_SPC", "OTC_NDX"]
    eu = ["OTC_FCHI", "OTC_GDAXI"]
    closes = {
        "OTC_SPC": [100.0, 95.0],
        "OTC_NDX": [100.0, 94.0],
        "OTC_FCHI": [100.0, 96.0],
        "OTC_GDAXI": [100.0, 95.5],
    }
    snap = build_macro_snapshot(us, eu, closes, {"min_indices_for_vote": 2})
    assert snap.tag == "risk_off"
    assert snap.eurusd_bias == "PUT"
    assert "MACRO_CONFLUENCIA" in snap.macro_block


def test_fx_reference_context_line_risk_on():
    line = fx_reference_context_line("risk_on", {"usdjpy": {"risk_on": "RISE", "risk_off": "FALL"}})
    assert "CONTEXTO_FX_REF" in line
    assert "Risk-On" in line
    assert "USDJPY RISE" in line


def test_fx_reference_divergence_eu_leads():
    line = fx_reference_context_line("divergence_eu_leads")
    assert "Divergencia EU lidera US" in line
    assert "USDJPY FALL" in line
    assert "AUDUSD RISE" in line


def test_format_macro_confluence_block_and_empty_snapshot():
    block = format_macro_confluence_block("US", "EU", "risk_on", "FX", eurusd_bias="CALL")
    assert "EURUSD_bias_quant=CALL" in block
    empty = empty_macro_snapshot()
    assert isinstance(empty, MacroSnapshot)
    assert empty.tag == "indefinido"


def test_resolve_macro_config_defaults_and_upper_labels():
    cfg = resolve_macro_config({"cluster_labels": {"us": ["s&p500"], "eu": ["dax40"]}})
    assert cfg["divergence_blocks_execution"] is True
    assert cfg["cluster_labels"]["us"] == ["S&P500"]
    assert cfg["cluster_labels"]["eu"] == ["DAX40"]
