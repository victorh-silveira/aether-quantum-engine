from unittest.mock import MagicMock

from src.application.services.llm.cluster_refresh_execute_policy import (
    cluster_refresh_may_execute,
    resolve_anchor,
)
from src.domain.models.trade import TradeDirection


def test_resolve_anchor_top_level_and_nested():
    assert resolve_anchor(None) == ""
    assert resolve_anchor({}) == ""
    assert resolve_anchor({"anchor": "frxEURUSD"}) == "frxEURUSD"
    assert resolve_anchor({"strategy": {"correlation": {"anchor": "frxGBPUSD"}}}) == "frxGBPUSD"


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


def test_refresh_resolves_anchor_from_config_when_missing():
    orch = _orch()
    orch.anchor = ""
    orch.config = {
        "anchor": "frxEURUSD",
        "strategy": {"correlation": {"anchor": "frxEURUSD"}},
        "orchestrator": orch.config["orchestrator"],
    }
    decisions = {
        "frxEURUSD": {"direction": TradeDirection.PUT, "metrics": {"macro_sentiment": "divergence_us_leads"}},
        "frxGBPUSD": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "macro_sentiment": "divergence_us_leads",
                "llm_statarb_dir_corrected": True,
                "cluster_target_sym": "frxGBPUSD",
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
        "frxEURUSD": {"direction": TradeDirection.PUT, "metrics": {"macro_sentiment": "divergence_us_leads"}},
        "frxGBPUSD": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "macro_sentiment": "divergence_us_leads",
                "llm_statarb_dir_corrected": True,
                "cluster_target_sym": "frxGBPUSD",
            },
        },
    }
    ok, _ = cluster_refresh_may_execute(orch, decisions, refresh_without_llm=True, now_epoch=2000.0)
    assert ok is True
