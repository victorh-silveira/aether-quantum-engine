from unittest.mock import MagicMock

from src.application.services.llm.llm_cluster_guards import cluster_execute_block_reason
from src.application.services.llm.profitable_scenario import (
    cluster_symbol_allowed_for_tag,
    min_conviction_for_macro_tag,
    resolve_profitable_scenario_config,
)
from src.domain.models.trade import TradeDirection


def test_cluster_symbol_open_without_allowlist():
    assert cluster_symbol_allowed_for_tag(None, macro_tag="risk_on", symbol="R_25") is True


def test_normalize_maps_skip_invalid_entries():
    cfg = resolve_profitable_scenario_config(
        {
            "allowed_cluster_symbols_by_tag": {"": ["R_50"], "risk_on": []},
            "min_conviction_by_tag": {"": 0.5, "risk_on": "bad", "risk_off": 0.68},
        }
    )
    assert "allowed_cluster_symbols_by_tag" not in cfg or "risk_on" not in (
        cfg.get("allowed_cluster_symbols_by_tag") or {}
    )
    assert cfg.get("min_conviction_by_tag", {}).get("risk_off") == 0.68


def test_resolve_profitable_scenario_config_merges_maps():
    cfg = resolve_profitable_scenario_config(
        {
            "allowed_cluster_symbols_by_tag": {"risk_off": ["R_75"]},
            "min_conviction_by_tag": {"risk_off": 0.68},
            "confluence_conviction_floor": 0.65,
        }
    )
    assert cfg["allowed_cluster_symbols_by_tag"]["risk_off"] == ("R_75",)
    assert cfg["min_conviction_by_tag"]["risk_off"] == 0.68


def test_cluster_symbol_allowlist_open_when_tag_missing():
    macro = {"allowed_cluster_symbols_by_tag": {"risk_on": ["R_50"]}}
    assert cluster_symbol_allowed_for_tag(macro, macro_tag="risk_off", symbol="R_75") is True


def test_cluster_symbol_allowlist_blocks_wrong_index():
    macro = {
        "allowed_cluster_symbols_by_tag": {"risk_on": ["R_50"]},
        "allowed_execute_tags": ("risk_on",),
        "confluence_conviction_floor": 0.65,
        "assert_min_hmm_prob": 0.0,
    }
    assert cluster_symbol_allowed_for_tag(macro, macro_tag="risk_on", symbol="R_25") is False
    assert cluster_symbol_allowed_for_tag(macro, macro_tag="risk_on", symbol="R_50") is True


def test_min_conviction_by_tag_raises_floor():
    macro = {"min_conviction_by_tag": {"risk_on": 0.72}}
    assert min_conviction_for_macro_tag(macro, macro_tag="risk_on", base_floor=0.60) == 0.72


def test_cluster_execute_blocks_scenario_symbol():
    orch = MagicMock()
    orch.config = {"llm": {"min_conviction_execute": 0.60}}
    orch._cluster_pause_after_loss_active = False
    macro = {
        "allowed_execute_tags": ("risk_on",),
        "allowed_cluster_symbols_by_tag": {"risk_on": ["R_50"]},
        "confluence_conviction_floor": 0.65,
        "assert_min_hmm_prob": 0.0,
    }
    metrics = {
        "macro_sentiment": "risk_on",
        "macro_us_strength_quant": 0.85,
        "macro_eu_strength_quant": 0.40,
        "hmm_prob": 0.9,
        "hmm_state": 0,
        "statarb_spreads": {"R_25": -2.0},
    }
    reason = cluster_execute_block_reason(
        orch,
        metrics,
        0.75,
        TradeDirection.CALL,
        macro,
        {"statarb_require_z_align": False},
        active_region="us",
        target_sym="R_25",
        llm_cluster_explicit=True,
    )
    assert reason == "scenario_symbol_not_allowed"
