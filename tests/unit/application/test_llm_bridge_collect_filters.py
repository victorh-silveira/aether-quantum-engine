from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.domain.models.trade import TradeDirection
from tests.unit.application.llm_response_fixtures import MOCK_LLM_CALL_LINE, MOCK_LLM_PUT_LINE


def _stub_closes(head: list[float]) -> list[float]:
    return head + [head[-1] * (1.001**i) for i in range(200)]


@pytest.mark.asyncio
async def test_collect_llm_decisions_preserva_conviccao_do_payload():
    orch = MagicMock()
    orch.config = {
        "llm": {
            "base_url": "http://x",
            "model": "m",
            "timeout_seconds": 5,
            "analysis_granularity_seconds": 300,
            "analysis_bars": 120,
            "ohlc_bars": 120,
            "min_conviction_execute": 0.67,
        },
        "risk_management": {"params": {"duration": 1, "duration_unit": "m", "payout_estimate": 0.95}},
    }
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    orch.stream.get_numpy_series = MagicMock(return_value=np.array([100.0 * (1.001**i) for i in range(120)]))
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=_stub_closes([99.0, 99.5, 100.0]))
    orch._neutral_metrics = MagicMock(return_value={"direction": "NONE"})
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("CALL", True, MOCK_LLM_CALL_LINE),
    ):
        out = await collect_llm_decisions(orch)
    assert out["frxEURUSD"]["direction"] == TradeDirection.CALL
    assert out["frxEURUSD"]["metrics"]["decision_source"] == "llm"
    assert out["frxEURUSD"]["metrics"]["execute"] is True
    assert out["frxEURUSD"]["metrics"]["conviction"] == pytest.approx(0.99, abs=1e-6)


@pytest.mark.asyncio
async def test_collect_llm_decisions_keeps_put_on_consecutive_cycles():
    orch = MagicMock()
    orch.config = {
        "llm": {
            "base_url": "http://x",
            "model": "m",
            "timeout_seconds": 5,
            "analysis_granularity_seconds": 300,
            "analysis_bars": 120,
            "m15_bars": 120,
            "m3_bars": 120,
        },
        "risk_management": {"params": {"duration": 1, "duration_unit": "m", "payout_estimate": 0.95}},
    }
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    orch.stream.get_numpy_series = MagicMock(return_value=np.array([100.0 * (1.001**i) for i in range(120)]))
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=_stub_closes([100.0, 99.0, 98.0, 97.0, 96.0]))
    orch._neutral_metrics = MagicMock(return_value={"direction": "NONE"})
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("PUT", True, MOCK_LLM_PUT_LINE),
    ):
        first = await collect_llm_decisions(orch)
        second = await collect_llm_decisions(orch)
    assert first["frxEURUSD"]["direction"] == TradeDirection.PUT
    assert second["frxEURUSD"]["direction"] == TradeDirection.PUT
    assert second["frxEURUSD"]["metrics"]["execute"] is True


@pytest.mark.asyncio
async def test_collect_llm_decisions_keeps_call_on_consecutive_cycles():
    orch = MagicMock()
    orch.config = {
        "llm": {
            "base_url": "http://x",
            "model": "m",
            "timeout_seconds": 5,
            "analysis_granularity_seconds": 300,
            "analysis_bars": 120,
            "m15_bars": 100,
            "m3_bars": 140,
        },
        "risk_management": {"params": {"duration": 1, "duration_unit": "m", "payout_estimate": 0.95}},
    }
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    orch.stream.get_numpy_series = MagicMock(return_value=np.array([100.0 * (1.001**i) for i in range(200)]))
    macro_s = [100.0 * (1.001**i) for i in range(120)]
    struct_s = [100.0 * (1.001**i) for i in range(120)]
    swing_s = [100.0 * (1.001**i) for i in range(120)]
    trig_s = [100.0 * (1.001**i) for i in range(120)]
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    micro_s = [100.0 * (1.001**i) for i in range(120)]
    orch.stream.fetch_candle_closes = AsyncMock(
        side_effect=[
            macro_s,
            struct_s,
            swing_s,
            micro_s,
            trig_s,
            macro_s,
            struct_s,
            swing_s,
            micro_s,
            trig_s,
            macro_s,
            struct_s,
            swing_s,
            micro_s,
            trig_s,
        ]
    )
    orch._neutral_metrics = MagicMock(return_value={"direction": "NONE"})
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("CALL", True, MOCK_LLM_CALL_LINE),
    ):
        first = await collect_llm_decisions(orch)
        second = await collect_llm_decisions(orch)
    assert first["frxEURUSD"]["direction"] == TradeDirection.CALL
    assert second["frxEURUSD"]["direction"] == TradeDirection.CALL
    assert second["frxEURUSD"]["metrics"]["execute"] is True


@pytest.mark.asyncio
async def test_collect_llm_decisions_wait_api_resolve_fallback():
    orch = MagicMock()
    orch.config = {
        "llm": {
            "base_url": "http://x",
            "model": "m",
            "timeout_seconds": 5,
            "analysis_granularity_seconds": 300,
            "analysis_bars": 120,
            "min_conviction_execute": 0.50,
        },
        "risk_management": {"params": {"duration": 1, "duration_unit": "m", "payout_estimate": 0.95}},
    }
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    orch.stream.get_numpy_series = MagicMock(return_value=np.array([100.0 * (1.001**i) for i in range(120)]))
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=_stub_closes([99.0, 99.5, 100.0]))
    orch._neutral_metrics = MagicMock(return_value={"direction": "NONE"})
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("WAIT", False, "WAIT"),
    ):
        out = await collect_llm_decisions(orch)
    assert out["frxEURUSD"]["direction"] is None
    assert "LLM_EURUSD_AUSENTE" in out["frxEURUSD"]["metrics"]["llm_note"]
    assert out["frxEURUSD"]["metrics"]["execute"] is False
