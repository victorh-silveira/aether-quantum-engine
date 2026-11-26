"""Regressoes de selecao de timeframe e fechamento de velas Laya."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from src.infrastructure.handlers.stream_handler import StreamHandler
from tests.unit.infrastructure.test_laya_decision_adapter import _sample_candles


@pytest.mark.parametrize("broker_epoch", [None, 1764001500, 1])
def test_closed_candles_exact_boundary_order_duplicates_and_clock(monkeypatch, broker_epoch):
    stream = StreamHandler(
        MagicMock(), ["R_10"], {"granularity": 86400, "micro_granularity": 300, "mini_granularity": 300}
    )
    bars = _sample_candles()
    now = bars[-1].epoch + 300
    monkeypatch.setattr("src.infrastructure.handlers.stream_handler.time.time", lambda: now)
    monkeypatch.setattr(stream.tick_buffer, "latest_tick_epoch", lambda symbol: broker_epoch)
    duplicate = replace(bars[0], close=999)
    stream.micro_candles["R_10"] = [*reversed(bars), duplicate, replace(bars[-1], epoch=now)]
    stream.macro_candles["R_10"] = [replace(bars[0], close=5000)]
    result = stream.get_closed_candles("R_10", 300)
    assert [bar.epoch for bar in result] == [bar.epoch for bar in bars]
    assert result[0].close == 999
    result.clear()
    assert stream.micro_candles["R_10"]
    assert stream.get_closed_candles("UNKNOWN", 300) == []
    with pytest.raises(ValueError, match="sem buffer"):
        stream.get_closed_candles("R_10", 60)
