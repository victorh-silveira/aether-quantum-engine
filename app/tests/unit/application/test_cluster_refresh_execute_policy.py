from unittest.mock import MagicMock

from src.application.services.llm.cluster_refresh_execute_policy import (
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
    orch.anchor = "frxEURUSD"
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


def test_resolve_cluster_refresh_policy_defaults_quant_tags():
    policy = resolve_cluster_refresh_policy({})
    assert policy["quant_tags"] == ("divergence_us_leads", "divergence_eu_leads")


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
            "cluster_target_sym": "OTC_DJI",
        },
    }
    assert entry_is_quant_validated(entry) is True


def test_entry_is_quant_validated_m5_aligned():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "execute": True,
            "cluster_target_sym": "OTC_DJI",
            "index_m5_dir_by_symbol": {"OTC_DJI": "down"},
        },
    }
    assert entry_is_quant_validated(entry) is True


def test_entry_is_quant_validated_rejects_bad_m5_map():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "execute": True,
            "cluster_target_sym": "OTC_DJI",
            "index_m5_dir_by_symbol": "bad",
        },
    }
    assert entry_is_quant_validated(entry) is False


def test_macro_tag_from_decisions_scans_cluster_entries():
    tag = macro_tag_from_decisions(
        {"OTC_DJI": {"metrics": {"macro_confluence_tag": "divergence_eu_leads"}}},
        "frxEURUSD",
    )
    assert tag == "divergence_eu_leads"


def test_macro_tag_from_decisions_skips_invalid_and_empty():
    tag = macro_tag_from_decisions(
        {"bad": "x", "OTC_DJI": {"metrics": {"macro_sentiment": ""}}},
        "frxEURUSD",
    )
    assert tag == ""


def test_any_quant_validated_cluster_entry_invalid_decisions():
    assert any_quant_validated_cluster_entry("bad", anchor_sym="frxEURUSD") is False


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
        "frxEURUSD": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "divergence_us_leads"}},
        "OTC_DJI": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "macro_sentiment": "divergence_us_leads",
                "llm_statarb_dir_corrected": True,
                "cluster_target_sym": "OTC_DJI",
            },
        },
    }
    ok, reason = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=1000.0)
    assert ok is True
    assert reason == "quant_refresh_ok"


def test_refresh_risk_on_without_llm_blocks():
    orch = _orch()
    decisions = {
        "frxEURUSD": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "risk_on"}},
        "OTC_DJI": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "macro_sentiment": "risk_on", "cluster_target_sym": "OTC_DJI"},
        },
    }
    ok, reason = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=1000.0)
    assert ok is False
    assert reason == "risk_regime_requires_fresh_llm"


def test_refresh_resolves_anchor_from_config_when_missing():
    orch = _orch()
    orch.anchor = ""
    orch.config = {
        "anchor": "R_100",
        "strategy": {"correlation": {"anchor": "R_100"}},
        "orchestrator": orch.config["orchestrator"],
    }
    decisions = {
        "R_100": {"direction": TradeDirection.PUT, "metrics": {"macro_sentiment": "divergence_us_leads"}},
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
    ok, _ = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=2000.0)
    assert ok is True


def test_refresh_uses_default_anchor_when_empty(monkeypatch):
    monkeypatch.setattr(
        "src.application.services.llm.cluster_refresh_execute_policy.resolve_anchor",
        lambda _cfg: "",
    )
    orch = _orch()
    orch.anchor = ""
    orch.config = {"orchestrator": orch.config["orchestrator"]}
    decisions = {
        "R_100": {"direction": TradeDirection.PUT, "metrics": {"macro_sentiment": "divergence_us_leads"}},
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
    ok, _ = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=2000.0)
    assert ok is True


def test_refresh_divergence_without_quant_edge_blocks():
    orch = _orch()
    decisions = {
        "frxEURUSD": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "divergence_us_leads"}},
        "OTC_DJI": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": True,
                "macro_sentiment": "divergence_us_leads",
                "cluster_target_sym": "OTC_DJI",
                "index_m5_dir_by_symbol": {"OTC_DJI": "down"},
            },
        },
    }
    ok, reason = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=1000.0)
    assert ok is False
    assert reason == "divergence_refresh_no_quant_edge"


def test_refresh_execute_disabled_on_divergence():
    orch = _orch(orchestrator={"cluster_refresh_execute_on_quant_validate": False})
    decisions = {
        "OTC_DJI": {
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
        "frxEURUSD": {"direction": TradeDirection.CALL, "metrics": {"macro_sentiment": "divergence_us_leads"}},
        "OTC_DJI": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "macro_sentiment": "divergence_us_leads",
                "llm_statarb_dir_corrected": True,
                "cluster_target_sym": "OTC_DJI",
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
    orch = _orch(last_cluster_cycle_end=900.0)
    ok, reason = cluster_entry_spacing_allows(orch, now_epoch=1000.0)
    assert ok is True
    assert reason == ""
