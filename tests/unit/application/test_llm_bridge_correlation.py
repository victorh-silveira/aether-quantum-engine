from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_collect_llm_decisions_with_correlation_enabled():
    """Verifica se a correlação propaga o sinal da âncora para os alvos."""
    orch = MagicMock()
    orch.anchor = "OTC_SPC"
    orch.symbols = ["OTC_SPC", "OTC_NDX", "OTC_DJI"]
    orch._active_cycle_id = 1
    orch.config = {
        "strategy": {
            "correlation": {
                "enabled": True,
                "targets": {
                    "OTC_NDX": 0.96,
                    "OTC_DJI": -0.5,  # Inversão para teste
                },
            }
        },
        "llm": {"max_decision_latency_seconds": 10},
    }

    metrics_anchor = {"conviction": 0.8, "direction": "CALL", "execute": True, "llm_note": "Strong trend"}

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (TradeDirection.CALL, metrics_anchor)

        decisions = await collect_llm_decisions(orch)

        assert mock_dec.call_count == 1

        assert decisions["OTC_SPC"]["direction"] == TradeDirection.CALL

        assert decisions["OTC_NDX"]["direction"] == TradeDirection.CALL
        assert decisions["OTC_NDX"]["metrics"]["decision_source"] == "cluster_regime"
        assert "CLUSTER (CALL)" in decisions["OTC_NDX"]["metrics"]["llm_note"]

        assert decisions["OTC_DJI"]["direction"] == TradeDirection.PUT
        assert decisions["OTC_DJI"]["metrics"]["decision_source"] == "cluster_regime"
        assert "CLUSTER (PUT)" in decisions["OTC_DJI"]["metrics"]["llm_note"]


@pytest.mark.asyncio
async def test_collect_llm_decisions_correlation_no_direction():
    """Verifica se não propaga nada se a âncora não tiver direção."""
    orch = MagicMock()
    orch.anchor = "OTC_SPC"
    orch.symbols = ["OTC_SPC", "OTC_NDX"]
    orch._active_cycle_id = 1
    orch.config = {
        "strategy": {"correlation": {"enabled": True, "targets": {"OTC_NDX": 0.96}}},
        "llm": {"max_decision_latency_seconds": 10},
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (None, {"execute": False})

        decisions = await collect_llm_decisions(orch)

        assert "OTC_SPC" in decisions
        assert "OTC_NDX" not in decisions  # Não deve ter decisão para o alvo


@pytest.mark.asyncio
async def test_collect_llm_decisions_global_inversion():
    """Verifica se o sinal da âncora é invertido globalmente antes da propagação."""
    orch = MagicMock()
    orch.anchor = "OTC_SPC"
    orch.symbols = ["OTC_SPC", "OTC_NDX"]
    orch._active_cycle_id = 2
    orch.config = {
        "llm": {"invert_llm_direction": True, "max_decision_latency_seconds": 10},
        "strategy": {"correlation": {"enabled": True, "targets": {"OTC_NDX": 1.0}}},
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (TradeDirection.CALL, {"conviction": 0.9, "llm_note": "Bullish"})

        decisions = await collect_llm_decisions(orch)

        assert decisions["OTC_SPC"]["direction"] == TradeDirection.PUT
        assert "INVERTED" in decisions["OTC_SPC"]["metrics"]["llm_note"]

        assert decisions["OTC_NDX"]["direction"] == TradeDirection.PUT
