from unittest.mock import MagicMock

from src.application.services.llm.cluster_refresh_execute_policy import (
    any_cluster_entry_marked_execute,
    any_cluster_entry_with_direction,
    any_quant_validated_cluster_entry,
    cluster_entry_spacing_allows,
    cluster_refresh_may_execute,
    entry_is_quant_validated,
    macro_tag_from_decisions,
    resolve_cluster_refresh_policy,
)
from src.domain.models.trade import TradeDirection


def _orch(**overrides):
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.config = {
        "orchestrator": {
            "cluster_refresh_execute_enabled": False,
            "cluster_refresh_execute_on_quant_validate": True,
            "cluster_refresh_quant_tags": ["divergence_us_leads", "divergence_eu_leads"],
            "post_settlement_breath_seconds": 60,
            "cluster_refresh_entry_spacing_seconds": 30,
            **overrides.get("orchestrator", {}),
        }
    }
    orch.state.active_contracts = overrides.get("active_contracts", {})
    orch._last_cluster_cycle_end = overrides.get("last_cluster_cycle_end", 0.0)
    return orch


def test_any_cluster_entry_with_direction_rejects_non_dict():
    assert any_cluster_entry_with_direction("bad", anchor_sym="R_100") is False


def test_resolve_cluster_refresh_policy_defaults_quant_tags():
    policy = resolve_cluster_refresh_policy({})
    assert policy["quant_tags"] == ("risk_on", "risk_off", "divergence_us_leads", "divergence_eu_leads")


def test_entry_is_quant_validated_rejects_invalid_entry():
    assert entry_is_quant_validated("bad") is False
    assert entry_is_quant_validated({"metrics": "x"}) is False
    assert entry_is_quant_validated({"metrics": {"execute": True}, "direction": None}) is False


def test_entry_is_quant_validated_statarb_source():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "decision_source": "cluster_statarb_dir",
            "cluster_target_sym": "R_50",
        },
    }
    assert entry_is_quant_validated(entry) is True


def test_entry_is_quant_validated_m5_aligned():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "execute": True,
            "cluster_target_sym": "R_50",
            "index_m5_dir_by_symbol": {"R_50": "down"},
        },
    }
    assert entry_is_quant_validated(entry) is True


def test_entry_is_quant_validated_rejects_bad_m5_map():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "execute": True,
            "cluster_target_sym": "R_50",
            "index_m5_dir_by_symbol": "bad",
        },
    }
    assert entry_is_quant_validated(entry) is False


def test_macro_tag_from_decisions_scans_cluster_entries():
    tag = macro_tag_from_decisions(
        {"R_50": {"metrics": {"macro_confluence_tag": "divergence_eu_leads"}}},
        "R_100",
    )
    assert tag == "divergence_eu_leads"


def test_macro_tag_from_decisions_skips_invalid_and_empty():
    tag = macro_tag_from_decisions(
        {"bad": "x", "R_50": {"metrics": {"macro_sentiment": ""}}},
        "R_100",
    )
    assert tag == ""


def test_any_cluster_entry_helpers():
    assert any_quant_validated_cluster_entry("bad", anchor_sym="R_100") is False
    assert any_cluster_entry_marked_execute("bad", anchor_sym="R_100") is False
    decisions = {
        "R_100": {"direction": TradeDirection.CALL, "metrics": {"execute": False}},
        "1HZ100V": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "macro_sentiment": "divergence_eu_leads"},
        },
    }
    assert any_cluster_entry_marked_execute(decisions, anchor_sym="R_100") is True


def test_refresh_not_required_when_llm_fresh():
    orch = _orch()
    ok, reason = cluster_refresh_may_execute(orch, {}, refresh_without_llm=False)
    assert ok is True
    assert reason == ""


def test_refresh_global_execute_enabled():
    orch = _orch(orchestrator={"cluster_refresh_execute_enabled": True})
    ok, reason = cluster_refresh_may_execute(orch, {}, refresh_without_llm=True)
    assert ok is True
    assert reason == "global_refresh_execute"


def test_refresh_divergence_quant_validated_allows_execute():
    orch = _orch()
    decisions = {
        "R_100": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "divergence_us_leads"}},
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "macro_sentiment": "divergence_us_leads",
                "llm_statarb_dir_corrected": True,
                "cluster_target_sym": "R_50",
            },
        },
    }
    ok, reason = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=1000.0)
    assert ok is True
    assert reason == "quant_refresh_ok"


def test_refresh_risk_off_with_quant_correction_allows():
    orch = _orch(
        orchestrator={
            "cluster_refresh_quant_tags": ["risk_on", "risk_off", "divergence_us_leads", "divergence_eu_leads"]
        }
    )
    decisions = {
        "R_100": {"direction": TradeDirection.PUT, "metrics": {"macro_sentiment": "risk_off"}},
        "1HZ100V": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": True,
                "macro_sentiment": "risk_off",
                "llm_statarb_dir_corrected": True,
                "cluster_target_sym": "1HZ100V",
            },
        },
    }
    ok, reason = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=2000.0)
    assert ok is True
    assert reason == "quant_refresh_ok"


def test_refresh_risk_on_without_llm_blocks():
    orch = _orch()
    decisions = {
        "R_100": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "risk_on"}},
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "macro_sentiment": "risk_on", "cluster_target_sym": "R_50"},
        },
    }
    ok, reason = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=1000.0)
    assert ok is False
    assert reason == "risk_regime_requires_fresh_llm"


def test_refresh_divergence_without_cluster_direction_blocks():
    orch = _orch()
    decisions = {
        "R_100": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "divergence_us_leads"}},
        "R_50": {
            "metrics": {
                "execute": False,
                "macro_sentiment": "divergence_us_leads",
                "cluster_target_sym": "R_50",
            },
        },
    }
    ok, reason = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=1000.0)
    assert ok is False
    assert reason == "divergence_refresh_no_quant_edge"


def test_refresh_divergence_with_cached_direction_allows():
    orch = _orch()
    decisions = {
        "R_100": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "divergence_us_leads"}},
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": False,
                "macro_sentiment": "divergence_us_leads",
                "cluster_target_sym": "R_50",
            },
        },
    }
    ok, reason = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=1000.0)
    assert ok is True
    assert reason == "quant_refresh_ok"


def test_refresh_execute_disabled_on_divergence():
    orch = _orch(orchestrator={"cluster_refresh_execute_on_quant_validate": False})
    decisions = {
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "macro_sentiment": "divergence_us_leads",
                "llm_statarb_dir_corrected": True,
            },
        },
    }
    ok, reason = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True)
    assert ok is False
    assert reason == "cluster_refresh_execute_disabled"


def test_refresh_spacing_blocks_after_settlement():
    orch = _orch(last_cluster_cycle_end=990.0)
    decisions = {
        "R_100": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "divergence_us_leads"}},
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "macro_sentiment": "divergence_us_leads",
                "llm_statarb_dir_corrected": True,
                "cluster_target_sym": "R_50",
            },
        },
    }
    ok, reason = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=1000.0)
    assert ok is False
    assert reason == "post_settlement_spacing"


def test_entry_spacing_blocks_active_contract():
    orch = _orch(active_contracts={"1": {}})
    ok, reason = cluster_entry_spacing_allows(orch, now_epoch=1000.0)
    assert ok is False
    assert reason == "active_contract_open"


def test_refresh_spacing_allows_after_m1_settlement_window():
    orch = _orch(
        last_cluster_cycle_end=990.0,
        orchestrator={
            "entry_spacing_follows_contract": True,
            "post_settlement_breath_seconds": 8,
            "cluster_refresh_entry_spacing_seconds": 5,
        },
    )
    orch.config = {
        "orchestrator": orch.config["orchestrator"],
        "risk_management": {"params": {"duration": 1, "duration_unit": "m"}},
    }
    ok, reason = cluster_entry_spacing_allows(orch, now_epoch=1004.0)
    assert ok is True
    assert reason == ""


def test_entry_spacing_zero_disables_wait():
    orch = _orch(
        orchestrator={
            "post_settlement_breath_seconds": 0,
            "cluster_refresh_entry_spacing_seconds": 0,
        },
        last_cluster_cycle_end=990.0,
    )
    ok, reason = cluster_entry_spacing_allows(orch, now_epoch=1000.0)
    assert ok is True
    assert reason == ""


def test_entry_spacing_allows_after_breath():
    ok, reason = cluster_entry_spacing_allows(_orch(last_cluster_cycle_end=900.0), now_epoch=1000.0)
    assert ok is True and reason == ""
