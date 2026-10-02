"""Testes de sincronizacao inicial de historico de stream."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.models.market_data import Candle
from src.infrastructure.handlers.stream_handler import StreamHandler
from src.infrastructure.handlers.stream_sync_start import sync_triple_candle_history


@pytest.mark.asyncio
async def test_stream_sync_start_macro_count_zero():
    """Valida espelhamento de micro para macro quando macro_count e zero."""
    ws = MagicMock()
    ws.is_running = True
    ws.subscribe = MagicMock()
    ws.send = AsyncMock(return_value={"candles": []})
    config = {
        "buffer_limit": 10,
        "fetch_count": 0,
        "granularity": 86400,
        "micro_granularity": 300,
        "mini_granularity": 300,
    }
    sh = StreamHandler(ws, ["R_10"], config)
    sh.micro_candles["R_10"] = [Candle("R_10", 1.0, 1.1, 0.9, 1.05, datetime.now(), 1000)]
    with patch(
        "src.infrastructure.handlers.stream_sync_start._resolve_sync_targets",
        return_value=(0, 0, 0),
    ):
        await sync_triple_candle_history(sh, AsyncMock())
    assert len(sh.macro_candles["R_10"]) == 1
    assert sh.macro_candles["R_10"][0].close == 1.05
