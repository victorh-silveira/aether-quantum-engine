from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.llm import llm_bridge as bridge
from src.application.services.llm.indicators import resolve_indicator_config
from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_request_payload_empty_model_response_returns_none():
    orch = MagicMock()
    runtime = {
        "base_url": "http://x",
        "model": "m",
        "timeout": 1.0,
        "num_predict": 64,
        "parse_retry_attempts": 0,
        "llm_async_outer_seconds": 30.0,
    }
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("", False, ""),
    ) as gen:
        payload = await bridge._request_payload(orch, "frxEURUSD", runtime, "p")
    gen.assert_awaited_once()
    assert payload["_direction_normalized"] is None


@pytest.mark.asyncio
async def test_request_payload_transporta_token_soberano():
    orch = MagicMock()
    runtime = {
        "base_url": "http://x",
        "model": "m",
        "timeout": 1.0,
        "num_predict": 64,
        "parse_retry_attempts": 1,
        "llm_async_outer_seconds": 30.0,
    }
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("CALL", True, "CALL"),
    ):
        payload = await bridge._request_payload(orch, "frxEURUSD", runtime, "p")
    assert payload["_direction_normalized"] == "CALL"


@pytest.mark.asyncio
async def test_collect_llm_decisions_keeps_execution_even_when_payout_is_low():
    orch = MagicMock()
    orch.config = {
        "llm": {
            "base_url": "http://x",
            "model": "m",
            "timeout_seconds": 5,
            "analysis_bars": 120,
            "ohlc_bars": 120,
        },
        "risk_management": {"params": {"duration": 1, "duration_unit": "m", "payout_estimate": 0.5}},
    }
    orch.symbols = ["frxEURUSD"]
    orch.anchor = "frxEURUSD"
    series = np.array([100.0 * (1.001**i) for i in range(120)])
    orch.stream.get_numpy_series = MagicMock(return_value=series)
    orch._neutral_metrics = MagicMock(return_value={"direction": None})
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=list(series))

    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("CALL", True, "CALL"),
    ):
        out = await collect_llm_decisions(orch)
    assert out["frxEURUSD"]["metrics"]["execute"] is True


@pytest.mark.asyncio
async def test_collect_symbol_decision_choppy_executes_as_ordered():
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.logger = MagicMock()
    orch.risk_manager = MagicMock()
    orch.risk_manager.get_wr_rolling_stats = MagicMock(return_value=(None, 0))
    ic = resolve_indicator_config({})
    runtime = {
        "base_url": "http://x",
        "model": "m",
        "timeout": 5.0,
        "num_predict": 64,
        "payout_estimate": 0.95,
        "min_payout_accept": 0.8,
        "duration": 1,
        "du": "m",
        "min_conviction_execute": 0.66,
        "logic_line_max_chars": 140,
        "indicator_config": ic,
        "max_decision_latency_seconds": 30.0,
    }
    extra = {
        "regime_label": "choppy",
        "atr_m5_pct": 1.5,
        "m15_closes": [100.0] * 120,
        "m5_closes": [100.0] * 120,
        "m3_closes": [100.0] * 120,
    }
    with (
        patch(
            "src.application.services.llm.symbol_decision_utils.fetch_context_blocks",
            new=AsyncMock(return_value=("m", "e", "s", "g", "mtf", extra)),
        ),
        patch(
            "src.application.services.llm.llm_symbol_io.get_decision",
            new=AsyncMock(return_value=("CALL", True, "CALL")),
        ),
    ):
        direction, metrics = await bridge._collect_symbol_decision(orch, sym="frxEURUSD", runtime=runtime)

    assert direction == TradeDirection.CALL
    assert metrics.get("execute") is True
