from src.application.services.llm.global_macro_confluence import (
    MacroSnapshot,
    aggregate_cluster_vote,
    build_macro_snapshot,
    classify_transatlantic_confluence,
    cluster_direction_from_closes,
    cluster_trade_direction,
    empty_macro_snapshot,
    eurusd_bias_from_confluence,
    expected_cluster_tags_line,
    format_macro_confluence_block,
    fx_reference_context_line,
    regional_intelligence_line,
    resolve_macro_config,
)


def test_cluster_direction_from_closes_thresholds():
    assert cluster_direction_from_closes([100.0, 103.0], 0.02) == "up"
    assert cluster_direction_from_closes([100.0, 97.0], 0.02) == "down"
    assert cluster_direction_from_closes([100.0, 100.01], 0.02) == "flat"
    assert cluster_direction_from_closes([100.0, 99.92], 0.02, 0.10) == "flat"
    assert cluster_direction_from_closes([100.0, 98.5], 0.02, 0.10) == "down"
    assert cluster_direction_from_closes([100.0, 100.03], 0.05, 0.10) == "flat"
    assert cluster_direction_from_closes([100.0, 100.04], 0.05, 0.03) == "flat"
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
    assert eurusd_bias_from_confluence("divergence_us_leads", us_dir="up", eu_dir="down") == "CALL"
    assert eurusd_bias_from_confluence("divergence_us_leads", us_dir="down", eu_dir="up") == "PUT"
    assert eurusd_bias_from_confluence("divergence_eu_leads", us_dir="down", eu_dir="up") == "CALL"
    assert eurusd_bias_from_confluence("divergence_eu_leads", us_dir="up", eu_dir="down") == "PUT"
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


def test_aggregate_cluster_vote_na_when_no_closes():
    vote = aggregate_cluster_vote(
        ["OTC_SPC"],
        {"OTC_SPC": [0.0, 100.0]},
        threshold_pct=0.02,
        min_indices=1,
        min_move_pct=0.01,
    )
    assert vote.direction == "flat"
    assert "N/A" in vote.parts[0]


def test_aggregate_cluster_vote_insufficient_indices_for_min():
    vote = aggregate_cluster_vote(
        ["OTC_SPC", "OTC_NDX"],
        {"OTC_SPC": [100.0, 105.0], "OTC_NDX": [100.0, 104.0]},
        threshold_pct=0.02,
        min_indices=3,
        min_move_pct=0.01,
    )
    assert vote.direction == "flat"


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
    assert "US_INTEL" in snap.macro_block
    assert "EU_INTEL" in snap.macro_block


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


def test_regional_intelligence_line_divergence_leads():
    eu_snap = MacroSnapshot(
        us_dir="down",
        eu_dir="up",
        us_strength=0.4,
        eu_strength=0.8,
        tag="divergence_eu_leads",
        eurusd_bias="CALL",
        cluster_status="ok",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
    )
    eu_line = regional_intelligence_line(eu_snap)
    assert "LEAD=EU" in eu_line
    us_snap = MacroSnapshot(
        us_dir="up",
        eu_dir="down",
        us_strength=0.8,
        eu_strength=0.4,
        tag="divergence_us_leads",
        eurusd_bias="CALL",
        cluster_status="ok",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
    )
    us_line = regional_intelligence_line(us_snap)
    assert "LEAD=US" in us_line
    bal_snap = MacroSnapshot(
        us_dir="up",
        eu_dir="up",
        us_strength=0.8,
        eu_strength=0.8,
        tag="risk_on",
        eurusd_bias="CALL",
        cluster_status="ok",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
    )
    assert "LEAD=BAL" in regional_intelligence_line(bal_snap)


def test_format_macro_confluence_block_and_empty_snapshot():
    block = format_macro_confluence_block("US", "EU", "risk_on", "FX", eurusd_bias="CALL")
    assert "EURUSD_bias_quant=CALL" in block
    empty = empty_macro_snapshot()
    assert isinstance(empty, MacroSnapshot)
    assert empty.tag == "indefinido"


def test_resolve_macro_config_defaults_and_upper_labels():
    cfg = resolve_macro_config({"cluster_labels": {"us": ["s&p500"], "eu": ["dax40"]}})
    assert cfg["statarb_z_threshold"] == 2.5
    assert cfg["cluster_min_move_pct"] == 0.06
    assert cfg["cluster_use_m5_fallback_when_flat"] is True
    assert cfg["cluster_labels"]["us"] == ["S&P500"]
    assert cfg["cluster_labels"]["eu"] == ["DAX40"]
    assert cfg["cluster_fallback_granularity_seconds"] == 300


def test_resolve_macro_config_normalizes_invalid_label_regions():
    cfg = resolve_macro_config({"cluster_labels": {"us": "bad", "xx": [1, 2]}})
    assert "us" not in cfg["cluster_labels"]
    assert cfg["cluster_labels"]["xx"] == ["1", "2"]


def test_expected_cluster_tags_line_risk_on():
    line = expected_cluster_tags_line(
        tag="risk_on",
        us_dir="up",
        eu_dir="up",
        us_strength=1.0,
        eu_strength=1.0,
    )
    assert "US_CLUSTER=CALL" in line
    assert "EU_CLUSTER=CALL" in line


def test_expected_cluster_tags_line_risk_off_and_partial():
    off = expected_cluster_tags_line(
        tag="risk_off",
        us_dir="down",
        eu_dir="down",
        us_strength=0.8,
        eu_strength=0.8,
        macro_cfg={"confluence_conviction_floor": 0.55},
    )
    assert "US_CLUSTER=PUT" in off
    us_only = expected_cluster_tags_line(
        tag="indefinido",
        us_dir="up",
        eu_dir="flat",
        us_strength=0.7,
        eu_strength=0.1,
        macro_cfg={"confluence_conviction_floor": 0.55},
    )
    assert "US_CLUSTER=CALL" in us_only
    eu_only = expected_cluster_tags_line(
        tag="indefinido",
        us_dir="flat",
        eu_dir="down",
        us_strength=0.1,
        eu_strength=0.7,
        macro_cfg={"confluence_conviction_floor": 0.55},
    )
    assert "EU_CLUSTER=PUT" in eu_only
