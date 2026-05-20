from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_collect_llm_decisions_propagates_cluster_tags_only():
    """Indices seguem US_CLUSTER e EU_CLUSTER da LLM, sem inversao por coeficiente."""
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_SPC", "OTC_NDX", "OTC_FCHI"]
    orch._active_cycle_id = 1
    orch.config = {
        "strategy": {
            "correlation": {"enabled": True},
            "clusters": {
                "us": ["OTC_SPC", "OTC_NDX"],
                "eu": ["OTC_FCHI"],
            },
        },
        "llm": {"max_decision_latency_seconds": 10},
    }

    metrics_anchor = {
        "conviction": 0.8,
        "direction": "CALL",
        "execute": True,
        "llm_note": "Strong trend",
        "us_cluster": "PUT",
        "eu_cluster": "PUT",
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (TradeDirection.CALL, metrics_anchor)

        decisions = await collect_llm_decisions(orch)

        assert mock_dec.call_count == 1
        assert decisions["frxEURUSD"]["direction"] == TradeDirection.CALL
        assert decisions["OTC_SPC"]["direction"] == TradeDirection.PUT
        assert decisions["OTC_NDX"]["direction"] == TradeDirection.PUT
        assert decisions["OTC_FCHI"]["direction"] == TradeDirection.PUT
        assert decisions["OTC_SPC"]["metrics"]["decision_source"] == "cluster_regime"


@pytest.mark.asyncio
async def test_collect_llm_decisions_correlation_no_direction():
    """Verifica se nao propaga nada se a ancora nao tiver direcao."""
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_NDX"]
    orch._active_cycle_id = 1
    orch.config = {
        "strategy": {"correlation": {"enabled": True}},
        "llm": {"max_decision_latency_seconds": 10},
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (None, {"execute": False})

        decisions = await collect_llm_decisions(orch)

        assert "frxEURUSD" in decisions
        assert "OTC_NDX" not in decisions


@pytest.mark.asyncio
async def test_collect_llm_decisions_global_inversion_does_not_flip_cluster_tags():
    """Inversao global afeta apenas a ancora; clusters mantem tags da LLM."""
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_SPC", "OTC_NDX"]
    orch._active_cycle_id = 2
    orch.config = {
        "llm": {"invert_llm_direction": True, "max_decision_latency_seconds": 10},
        "strategy": {
            "correlation": {"enabled": True},
            "clusters": {"us": ["OTC_SPC", "OTC_NDX"], "eu": []},
        },
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (
            TradeDirection.CALL,
            {
                "conviction": 0.9,
                "llm_note": "Bullish",
                "us_cluster": "PUT",
                "eu_cluster": "PUT",
            },
        )

        decisions = await collect_llm_decisions(orch)

        assert decisions["frxEURUSD"]["direction"] == TradeDirection.PUT
        assert "INVERTED" in decisions["frxEURUSD"]["metrics"]["llm_note"]
        assert decisions["OTC_SPC"]["direction"] == TradeDirection.PUT
        assert decisions["OTC_NDX"]["direction"] == TradeDirection.PUT


@pytest.mark.asyncio
async def test_collect_llm_decisions_skips_symbols_outside_cluster_lists():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_SSMI", "OTC_SPC"]
    orch._active_cycle_id = 3
    orch.config = {
        "strategy": {
            "correlation": {"enabled": True},
            "clusters": {"us": ["OTC_SPC"], "eu": []},
        },
        "llm": {"max_decision_latency_seconds": 10},
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (
            TradeDirection.PUT,
            {"conviction": 0.8, "us_cluster": "PUT", "eu_cluster": "PUT"},
        )
        decisions = await collect_llm_decisions(orch)

    assert "OTC_SPC" in decisions
    assert "OTC_SSMI" not in decisions
