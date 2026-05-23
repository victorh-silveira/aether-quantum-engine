import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.application.services.llm.macro_config import MacroSnapshot
from src.domain.models.trade import TradeDirection
from tests.unit.application.llm_response_fixtures import MOCK_LLM_CALL_LINE
from tests.unit.application.macro_guard_fixtures import merge_orch_config


@pytest.mark.asyncio
async def test_collect_llm_decisions_success():
    orch = MagicMock()
    orch.config = merge_orch_config(
        {
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
    )
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    orch.stream.get_numpy_series = MagicMock(return_value=np.linspace(100.0, 101.0, 10))
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=[99.0, 99.5, 100.0])
    orch._neutral_metrics = MagicMock(return_value={"direction": None})

    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("CALL", True, MOCK_LLM_CALL_LINE),
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


@pytest.mark.asyncio
async def test_collect_llm_decisions_skips_llm_when_macro_tag_blocked():
    snap = MacroSnapshot(
        us_dir="up",
        eu_dir="up",
        us_strength=0.9,
        eu_strength=0.88,
        tag="risk_on",
        eurusd_bias="CALL",
        cluster_status="",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
    )
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch._active_cycle_id = 1
    orch.config = {
        "strategy": {"macro": {"allowed_execute_tags": ["risk_off"]}},
        "llm": {"max_decision_latency_seconds": 5},
    }

    with patch(
        "src.application.services.llm.llm_bridge.fetch_macro_snapshot",
        new_callable=AsyncMock,
        return_value=snap,
    ):
        out = await collect_llm_decisions(orch)

    assert out["frxEURUSD"]["direction"] is None
    assert "MACRO_SKIP" in out["frxEURUSD"]["metrics"]["llm_note"]


@pytest.mark.asyncio
async def test_collect_llm_decisions_reuses_cached_decisions_same_tag():
    snap = MacroSnapshot(
        us_dir="down",
        eu_dir="down",
        us_strength=0.9,
        eu_strength=0.88,
        tag="risk_off",
        eurusd_bias="PUT",
        cluster_status="",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
    )
    cached = {
        "frxEURUSD": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "decision_source": "llm"},
        }
    }
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch._active_cycle_id = 2
    orch._last_llm_macro_tag = "risk_off"
    orch._last_llm_decisions = dict(cached)
    orch.config = {"strategy": {"macro": {}}, "llm": {"max_decision_latency_seconds": 5}}

    with (
        patch(
            "src.application.services.llm.llm_bridge.fetch_macro_snapshot",
            new_callable=AsyncMock,
            return_value=snap,
        ),
        patch(
            "src.application.services.llm.llm_bridge._collect_symbol_decision",
            new_callable=AsyncMock,
        ) as mock_dec,
    ):
        out = await collect_llm_decisions(orch)
        mock_dec.assert_not_called()

    assert out["frxEURUSD"]["direction"] == TradeDirection.CALL
