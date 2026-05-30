from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.application.services.llm.llm_cluster_exclusive import (
    cluster_region_for_symbol,
    exclusive_cluster_by_macro_enabled,
    resolve_exclusive_cluster_region,
)
from src.domain.models.trade import TradeDirection


def test_resolve_exclusive_cluster_region_risk_on_off_and_divergence():
    assert resolve_exclusive_cluster_region({"macro_sentiment": "risk_on"}) == "us"
    assert resolve_exclusive_cluster_region({"macro_sentiment": "risk_off"}) == "eu"
    assert resolve_exclusive_cluster_region({"macro_sentiment": "divergence_us_leads"}) == "us"
    assert resolve_exclusive_cluster_region({"macro_sentiment": "divergence_eu_leads"}) == "eu"


def test_resolve_exclusive_cluster_region_unknown_tag_allows_both():
    assert resolve_exclusive_cluster_region({"macro_sentiment": "custom_tag"}) is None


def test_resolve_exclusive_cluster_region_indefinido_by_strength():
    assert (
        resolve_exclusive_cluster_region(
            {
                "macro_sentiment": "indefinido",
                "macro_us_strength_quant": 0.72,
                "macro_eu_strength_quant": 0.55,
            }
        )
        == "us"
    )
    assert (
        resolve_exclusive_cluster_region(
            {
                "macro_sentiment": "indefinido",
                "macro_us_strength_quant": 0.40,
                "macro_eu_strength_quant": 0.61,
            }
        )
        == "eu"
    )
    assert resolve_exclusive_cluster_region({"macro_sentiment": "indefinido"}) is None


def test_exclusive_cluster_config_sources():
    orch = MagicMock()
    orch.config = {"strategy": {"correlation": {"exclusive_cluster_by_macro": False}}}
    assert exclusive_cluster_by_macro_enabled(orch) is False
    orch.config = {"strategy": {"macro": {"exclusive_cluster_by_macro": True}}}
    assert exclusive_cluster_by_macro_enabled(orch) is True
    orch.config = {"strategy": {"correlation": {}, "macro": {}}}
    assert exclusive_cluster_by_macro_enabled(orch) is True


def test_cluster_region_for_symbol():
    assert cluster_region_for_symbol("R_25", us_targets=["R_25"], eu_targets=["R_75"]) == "us"
    assert cluster_region_for_symbol("R_75", us_targets=["R_25"], eu_targets=["R_75"]) == "eu"
    assert cluster_region_for_symbol("R_100", us_targets=["R_25"], eu_targets=["R_75"]) is None


@pytest.mark.asyncio
async def test_collect_llm_decisions_exclusive_risk_on_us_only():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_25", "R_75"]
    orch._active_cycle_id = 2
    orch.config = {
        "strategy": {
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": True,
                "cluster_invert_on_block": False,
            },
            "clusters": {"us": ["R_25"], "eu": ["R_75"]},
        },
        "llm": {"max_decision_latency_seconds": 10, "min_conviction_execute": 0.5},
    }

    with patch(
        "src.application.services.llm.llm_bridge._collect_symbol_decision",
        new_callable=AsyncMock,
    ) as mock_dec:
        mock_dec.return_value = (
            TradeDirection.CALL,
            {
                "conviction": 0.68,
                "us_cluster": "CALL",
                "eu_cluster": "PUT",
                "macro_sentiment": "risk_on",
            },
        )
        decisions = await collect_llm_decisions(orch)

    assert "R_25" in decisions
    assert decisions["R_25"]["direction"] == TradeDirection.CALL
    assert "R_75" not in decisions
