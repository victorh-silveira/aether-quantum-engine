from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.application.services.llm.llm_bridge_utils import parse_llm_trade_response
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_collect_llm_decisions_with_dynamic_cluster_regime():
    """Verifica se o robô segue as direções específicas dos clusters enviadas pela Gemini."""
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_SPC", "OTC_FCHI"]
    orch._active_cycle_id = 1
    orch.config = {
        "strategy": {
            "correlation": {
                "enabled": True,
                "targets": {"OTC_SPC": 1.0, "OTC_FCHI": 1.0},
            }
        },
        "llm": {"max_decision_latency_seconds": 10},
    }

    metrics_anchor = {
        "conviction": 0.8,
        "direction": "PUT",
        "execute": True,
        "llm_note": "Morphological Regime",
        "us_cluster": "PUT",
        "eu_cluster": "CALL",
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (TradeDirection.PUT, metrics_anchor)

        decisions = await collect_llm_decisions(orch)

        assert decisions["frxEURUSD"]["direction"] == TradeDirection.PUT

        assert decisions["OTC_SPC"]["direction"] == TradeDirection.PUT
        assert decisions["OTC_SPC"]["metrics"]["decision_source"] == "cluster_regime"

        assert decisions["OTC_FCHI"]["direction"] == TradeDirection.CALL
        assert decisions["OTC_FCHI"]["metrics"]["decision_source"] == "cluster_regime"
        assert "CLUSTER (CALL)" in decisions["OTC_FCHI"]["metrics"]["llm_note"]


def test_parse_llm_trade_response_with_clusters():
    """Valida o parsing das novas tags de cluster no texto da LLM."""
    raw_text = "EURUSD: CALL | US_CLUSTER: PUT | EU_CLUSTER: CALL | Probabilidade: 85%"
    out = parse_llm_trade_response(raw_text)

    assert out["direction"] == "CALL"
    assert out["us_cluster"] == "PUT"
    assert out["eu_cluster"] == "CALL"
    assert out["conviction"] == 0.85
    assert out["note"] == "EURUSD_CALL"
