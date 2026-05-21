"""Testes para normalização de decisões e tratamento de falhas da Gemini."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.llm import IndicatorConfig
from src.application.services.llm.llm_bridge import llm_metrics
from src.application.services.llm.llm_bridge_guards import is_sawtooth_pattern
from src.application.services.llm.llm_bridge_utils import strict_normalize_direction
from src.application.services.llm.symbol_decision import collect_symbol_llm_decision
from src.domain.models.trade import TradeDirection


def test_normalize_invalid_tokens():
    """Valida que tokens de espera ou invalidos normalizam para None."""
    assert strict_normalize_direction("SKIP 95%") is None
    assert strict_normalize_direction("SKIP 90%") is None
    assert strict_normalize_direction("WAIT") is None
    assert strict_normalize_direction("NEUTRAL") is None
    assert strict_normalize_direction("AGUARDAR") is None


def test_llm_metrics_failure_treatment():
    """Valida que resultados sem direcao sao atribuidos como falha de API."""
    metrics = llm_metrics(None, 0.95, "SKIP 95%")
    assert metrics["direction"] == "NONE"
    assert metrics["conviction"] == 0.0
    assert metrics["decision_source"] == "llm_skip"
    assert metrics["execute"] is False


def test_llm_metrics_real_failure():
    """Valida que falhas reais (sem token de espera) são atribuídas como falha de API."""
    metrics = llm_metrics(None, 0.0, "Erro interno")
    assert metrics["direction"] == "NONE"
    assert metrics["decision_source"] == "llm_api_failure"


def test_llm_metrics_put_coverage():
    """Aumenta cobertura para decisões de PUT."""
    metrics = llm_metrics(TradeDirection.PUT, 0.8, "Baixa confirmada")
    assert metrics["direction"] == "PUT"
    assert metrics["prob_put"] == 1.0
    assert metrics["prob_call"] == 0.0


def test_is_sawtooth_pattern_coverage():
    """Aumenta cobertura para o padrão sawtooth."""
    assert is_sawtooth_pattern("P/M/P/M") is True
    assert is_sawtooth_pattern("M/P/M/P") is True
    assert is_sawtooth_pattern("P/P/P/P") is False
    assert is_sawtooth_pattern("P/M") is False


@pytest.mark.asyncio
async def test_collect_symbol_llm_decision_missing_rm_and_ohlc(mocker):
    """Cobre ramos de RM ausente e OHLC ausente no orquestrador de decisão."""
    orch = MagicMock()
    orch.risk_manager = None
    orch._active_cycle_id = 1
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    orch.logger = MagicMock()

    mocker.patch(
        "src.application.services.llm.symbol_decision_utils.fetch_context_blocks",
        AsyncMock(return_value=("m", "st", "sw", "tr", "mtf", {"regime_label": "range"})),
    )

    mocker.patch(
        "src.application.services.llm.symbol_decision.request_llm_payload",
        AsyncMock(
            return_value={
                "_direction_normalized": "WAIT",
                "_conviction_normalized": 0.9,
                "_llm_direction_from_api": True,
            }
        ),
    )

    runtime = {
        "indicator_config": IndicatorConfig(),
        "payout_estimate": 0.85,
        "min_payout_accept": 0.80,
        "duration": 5,
        "du": "m",
        "model": "gemini",
        "base_url": "url",
        "timeout": 10,
        "num_predict": 10,
    }

    dir_final, metrics = await collect_symbol_llm_decision(
        orch, sym="frxEURUSD", runtime=runtime, llm_metrics=llm_metrics
    )
    assert dir_final is None
    assert "LLM_EURUSD_AUSENTE" in metrics["llm_note"]
