import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_collect_llm_decisions_success():
    orch = MagicMock()
    orch.config = {
        "llm": {
            "base_url": "http://x",
            "model": "m",
            "timeout_seconds": 5,
            "ohlc_bars": 4,
            "analysis_granularity_seconds": 300,
            "analysis_bars": 5,
        },
        "risk_management": {"params": {"duration": 1, "duration_unit": "m"}},
    }
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    orch.stream.get_numpy_series = MagicMock(return_value=np.linspace(100.0, 101.0, 10))
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=[99.0, 99.5, 100.0])
    orch._neutral_metrics = MagicMock(return_value={"direction": None})

    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("CALL", True, "CALL"),
    ):
        out = await collect_llm_decisions(orch)

    assert out["frxEURUSD"]["direction"] == TradeDirection.CALL
    assert out["frxEURUSD"]["metrics"]["execute"] is True
    assert out["frxEURUSD"]["metrics"]["decision_source"] == "llm"


@pytest.mark.asyncio
async def test_collect_llm_decisions_deadline_results_in_failure():
    orch = MagicMock()
    orch.config = {
        "llm": {
            "base_url": "http://x",
            "model": "m",
            "timeout_seconds": 0.1,
            "max_decision_latency_seconds": 0.01,
            "min_conviction_execute": 0.50,
        },
        "risk_management": {"params": {"duration": 1, "duration_unit": "m"}},
    }
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    orch._neutral_metrics = MagicMock(return_value={"direction": None})

    async def slow(*_a, **_k):
        await asyncio.sleep(2.0)
        return ("CALL", True, "CALL")

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", side_effect=TimeoutError()):
        out = await collect_llm_decisions(orch)

    assert out["frxEURUSD"]["direction"] is None
    assert out["frxEURUSD"]["metrics"]["execute"] is False
    assert "LLM Timeout" in out["frxEURUSD"]["metrics"]["llm_note"]


@pytest.mark.asyncio
async def test_collect_llm_decisions_api_error_results_in_failure():
    orch = MagicMock()
    orch.config = {
        "llm": {
            "base_url": "http://x",
            "model": "m",
            "timeout_seconds": 5,
            "min_conviction_execute": 0.50,
        },
        "risk_management": {"params": {"duration": 1, "duration_unit": "m"}},
    }
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    orch._neutral_metrics = MagicMock(return_value={"direction": None})
    orch.stream.get_numpy_series = MagicMock(return_value=np.linspace(100.0, 101.0, 10))
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=[99.0, 99.5, 100.0])

    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=(None, False, ""),
    ):
        out = await collect_llm_decisions(orch)

    assert out["frxEURUSD"]["direction"] is None
    assert out["frxEURUSD"]["metrics"]["execute"] is False


class _LlmTransportTestError(Exception):
    pass


@pytest.mark.asyncio
async def test_collect_llm_decisions_raises_on_critical_failure():
    orch = MagicMock()
    orch.config = {
        "llm": {"base_url": "http://x", "model": "m", "timeout_seconds": 1},
        "risk_management": {"params": {"duration": 1, "duration_unit": "t"}},
    }
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    orch.stream.get_numpy_series = MagicMock(return_value=np.array([1.0, 2.0]))
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=[])
    orch._neutral_metrics = MagicMock(return_value={})

    with (
        patch(
            "src.application.services.llm.llm_symbol_io.get_decision",
            new_callable=AsyncMock,
            side_effect=_LlmTransportTestError("critical down"),
        ),
        pytest.raises(_LlmTransportTestError, match="critical down"),
    ):
        await collect_llm_decisions(orch)
