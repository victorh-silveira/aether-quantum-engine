from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.application.services.llm import llm_bridge as bridge
from src.application.services.llm.indicators import resolve_indicator_config
from src.application.services.llm.llm_symbol_io import last_reference_price
from src.domain.models.trade import TradeDirection


def test_build_metrics_for_decision_nao_altera_lado_por_contradicao_narrativa():
    direction, metrics = bridge._build_metrics_for_decision_core(
        {"min_conviction_execute": 0.66, "model": "m"},
        TradeDirection.CALL,
        0.82,
        "nota forte",
        None,
        "M15: alta | M5: alta | M3: alta",
        "H1 alta",
        "M15 tendencia_EMA=baixa",
        "M5 tendencia_EMA=baixa",
        "M3 gatilho com tendencia_EMA=baixa",
        bridge.llm_metrics,
    )
    assert direction == TradeDirection.CALL
    assert metrics["direction"] == "CALL"
    assert metrics["llm_direction_adjusted"] is False


@pytest.mark.asyncio
async def test_fetch_context_blocks_returns_narrative_strings_and_context_extra():
    orch = MagicMock()
    orch.config = {"llm": {}}
    series = np.array([100.0 * (1.001**i) for i in range(120)])
    orch.stream.get_numpy_series = MagicMock(return_value=series)
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=[])
    orch.stream.fetch_candle_closes = AsyncMock(return_value=list(series))
    runtime = {
        "tf_trigger_gran": 300,
        "tf_trigger_bars": 120,
        "tf_structure_gran": 3600,
        "tf_structure_bars": 120,
        "tf_swing_gran": 900,
        "tf_swing_bars": 120,
        "tf_macro_gran": 14400,
        "tf_macro_bars": 120,
        "m3_max_ema_distance_pct": 0.8,
        "indicator_config": resolve_indicator_config({}),
    }
    macro_d, struct_d, swing_d, trig_d, mtf, extra = await bridge.fetch_context_blocks(orch, "1HZ75V", runtime)
    assert (
        isinstance(macro_d, str)
        and isinstance(struct_d, str)
        and isinstance(swing_d, str)
        and isinstance(trig_d, str)
        and isinstance(mtf, str)
    )
    assert isinstance(extra, dict)
    assert "sniper_tokens" in extra
    assert "llm_mtf_confluence_m30_m5" in extra


@pytest.mark.asyncio
async def test_fetch_context_blocks_usa_ohlc_m1_quando_disponivel():
    orch = MagicMock()
    orch.config = {"llm": {}}
    series = np.array([100.0 * (1.001**i) for i in range(120)])
    orch.stream.get_numpy_series = MagicMock(return_value=series)
    m1_closes = list(series)
    ohlc_rows = [(c, c + 0.6, c - 0.6, c) for c in m1_closes]
    orch.stream.fetch_candle_ohlc = AsyncMock(return_value=ohlc_rows)
    orch.stream.fetch_candle_closes = AsyncMock(return_value=m1_closes)
    runtime = {
        "tf_trigger_gran": 60,
        "tf_trigger_bars": 120,
        "tf_structure_gran": 3600,
        "tf_structure_bars": 120,
        "tf_swing_gran": 900,
        "tf_swing_bars": 120,
        "tf_macro_gran": 14400,
        "tf_macro_bars": 120,
        "m3_max_ema_distance_pct": 0.8,
        "indicator_config": resolve_indicator_config({}),
    }
    _macro, _struct, _swing, _trig, _mtf, extra = await bridge.fetch_context_blocks(orch, "1HZ75V", runtime)
    assert extra.get("m3_closes") == m1_closes[-120:]
    assert extra.get("sniper_tokens", {}).get("wick") != "na"


def test_last_reference_price_reads_close_and_empty():
    stream = MagicMock()
    stream.get_numpy_series.return_value = np.array([10.0, 10.5])
    assert last_reference_price(stream, "1HZ75V") == 10.5
    stream.get_numpy_series.return_value = np.array([])
    assert last_reference_price(stream, "1HZ75V") is None


def test_last_reference_price_on_error():
    stream = MagicMock()
    stream.get_numpy_series.side_effect = RuntimeError("x")
    assert last_reference_price(stream, "1HZ75V") is None
