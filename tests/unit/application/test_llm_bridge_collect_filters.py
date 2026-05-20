from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.domain.models.trade import TradeDirection


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
    orch.symbols = ["1HZ75V"]
    orch.anchor = "1HZ75V"
    orch.stream.get_numpy_series = MagicMock(return_value=np.array([100.0 * (1.001**i) for i in range(120)]))
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=_stub_closes([99.0, 99.5, 100.0]))
    orch._neutral_metrics = MagicMock(return_value={"direction": "NONE"})
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("CALL", True, "CALL"),
    ):
        out = await collect_llm_decisions(orch)
    assert out["1HZ75V"]["direction"] == TradeDirection.CALL
    assert out["1HZ75V"]["metrics"]["decision_source"] == "llm"
    assert out["1HZ75V"]["metrics"]["execute"] is True
    assert out["1HZ75V"]["metrics"]["conviction"] == pytest.approx(0.99, abs=1e-6)


@pytest.mark.asyncio
async def test_collect_llm_decisions_blocks_repeated_same_direction_streak():
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
            "wait_promotion_enabled": True,
            "wait_promotion_min_score": 6,
            "wait_promotion_min_conviction": 0.8,
            "max_same_direction_streak": 1,
            "neutral_weak_mtf_enabled": False,
        },
        "risk_management": {"params": {"duration": 1, "duration_unit": "m", "payout_estimate": 0.95}},
    }
    orch.symbols = ["1HZ75V"]
    orch.anchor = "1HZ75V"
    orch.stream.get_numpy_series = MagicMock(return_value=np.array([100.0 * (1.001**i) for i in range(120)]))
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=_stub_closes([100.0, 99.0, 98.0, 97.0, 96.0]))
    orch._neutral_metrics = MagicMock(return_value={"direction": "NONE"})
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("PUT", True, "PUT"),
    ):
        first = await collect_llm_decisions(orch)
        second = await collect_llm_decisions(orch)
    assert first["1HZ75V"]["direction"] == TradeDirection.PUT
    assert second["1HZ75V"]["direction"] == TradeDirection.PUT
    assert second["1HZ75V"]["metrics"]["execute"] is True


@pytest.mark.asyncio
async def test_collect_llm_decisions_blocks_repeated_direction_without_strict_confirmation():
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
            "same_direction_strict_enabled": True,
            "same_direction_rsi_min": 40,
            "same_direction_rsi_max": 60,
            "same_direction_require_m3_confirmation": True,
            "same_direction_require_wick_confirmation": True,
            "max_same_direction_streak": 3,
            "llm_laws_enabled": False,
            "rsi_exhaustion_gate_enabled": False,
        },
        "risk_management": {"params": {"duration": 1, "duration_unit": "m", "payout_estimate": 0.95}},
    }
    orch.symbols = ["1HZ75V"]
    orch.anchor = "1HZ75V"
    orch.stream.get_numpy_series = MagicMock(return_value=np.array([100.0 * (1.001**i) for i in range(200)]))
    macro_s = [100.0 * (1.001**i) for i in range(120)]
    struct_s = [100.0 * (1.001**i) for i in range(120)]
    swing_s = [100.0 * (1.001**i) for i in range(120)]
    trig_s = [100.0 * (1.001**i) for i in range(120)]
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(
        side_effect=[macro_s, struct_s, swing_s, trig_s, macro_s, struct_s, swing_s, trig_s]
    )
    orch._neutral_metrics = MagicMock(return_value={"direction": "NONE"})
    with patch(
        "src.application.services.llm.llm_symbol_io.get_decision",
        new_callable=AsyncMock,
        return_value=("CALL", True, "CALL"),
    ):
        first = await collect_llm_decisions(orch)
        second = await collect_llm_decisions(orch)
    assert first["1HZ75V"]["direction"] == TradeDirection.CALL
    assert second["1HZ75V"]["direction"] == TradeDirection.CALL
    assert second["1HZ75V"]["metrics"]["execute"] is True


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
    orch.symbols = ["1HZ75V"]
    orch.anchor = "1HZ75V"
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
    assert out["1HZ75V"]["direction"] is None
    assert "LLM Refused - Waiting" in out["1HZ75V"]["metrics"]["llm_note"]
    assert out["1HZ75V"]["metrics"]["execute"] is False
