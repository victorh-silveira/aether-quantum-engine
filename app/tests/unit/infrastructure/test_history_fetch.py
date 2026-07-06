import logging
from unittest.mock import AsyncMock, patch

import pytest

from src.domain.models.market_data import Candle
from src.infrastructure.handlers.history_fetch import (
    fetch_paginated_candle_history,
    is_rate_limit_error,
    merge_candle_pages,
    parse_history_fetch_config,
)


def test_is_rate_limit_error():
    assert is_rate_limit_error({"error": {"message": "You have reached the rate limit for ticks_history."}})
    assert not is_rate_limit_error({"error": {"message": "Invalid granularity"}})
    assert not is_rate_limit_error({})


def test_merge_candle_pages_empty_batch():
    existing = [Candle("RDBULL", 1, 1, 1, 1, None, 200)]
    assert merge_candle_pages(existing, []) == existing


def test_merge_candle_pages_deduplicates():
    older = [Candle("RDBULL", 1, 1, 1, 1, None, 100)]
    newer = [Candle("RDBULL", 1, 1, 1, 1, None, 200), Candle("RDBULL", 1, 1, 1, 1, None, 100)]
    merged = merge_candle_pages(newer, older)
    assert [c.epoch for c in merged] == [100, 200]


@pytest.mark.asyncio
async def test_fetch_paginated_retries_rate_limit():
    page = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "epoch": 2000}]
    ws = AsyncMock()
    ws.send = AsyncMock(
        side_effect=[
            {"error": {"message": "You have reached the rate limit for ticks_history."}},
            {"candles": page},
        ]
    )
    cfg = parse_history_fetch_config(
        {
            "history_fetch_chunk": 1,
            "history_fetch_delay_seconds": 0,
            "history_fetch_rate_limit_retries": 2,
        }
    )
    out = await fetch_paginated_candle_history(
        ws,
        symbol="RDBULL",
        granularity=60,
        target=1,
        fetch_cfg=cfg,
        logger=logging.getLogger("test"),
    )
    assert len(out) == 1
    assert ws.send.await_count == 2


@pytest.mark.asyncio
async def test_fetch_paginated_resumes_existing():
    existing = [Candle("RDBULL", 1, 1, 1, 1, None, 3000)]
    older = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.02, "epoch": 1000 + i} for i in range(2)]
    ws = AsyncMock()
    ws.send = AsyncMock(return_value={"candles": older})
    cfg = parse_history_fetch_config({"history_fetch_chunk": 10, "history_fetch_delay_seconds": 0})
    out = await fetch_paginated_candle_history(
        ws,
        symbol="RDBULL",
        granularity=60,
        target=3,
        fetch_cfg=cfg,
        logger=logging.getLogger("test"),
        existing=existing,
    )
    assert len(out) == 3
    assert out[0].epoch == 1000
    assert out[-1].epoch == 3000
    req = ws.send.await_args.args[0]
    assert req["end"] == 2999


@pytest.mark.asyncio
async def test_fetch_paginated_returns_when_target_already_met():
    existing = [Candle("RDBULL", 1, 1, 1, 1, None, 100 + i) for i in range(3)]
    ws = AsyncMock()
    cfg = parse_history_fetch_config({"history_fetch_delay_seconds": 0})
    out = await fetch_paginated_candle_history(
        ws,
        symbol="RDBULL",
        granularity=60,
        target=2,
        fetch_cfg=cfg,
        logger=logging.getLogger("test"),
        existing=existing,
    )
    assert len(out) == 2
    ws.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_paginated_stops_on_fatal_error():
    ws = AsyncMock()
    ws.send = AsyncMock(return_value={"error": {"message": "Invalid granularity"}})
    cfg = parse_history_fetch_config({"history_fetch_delay_seconds": 0})
    out = await fetch_paginated_candle_history(
        ws,
        symbol="RDBULL",
        granularity=60,
        target=5,
        fetch_cfg=cfg,
        logger=logging.getLogger("test"),
    )
    assert out == []


@pytest.mark.asyncio
async def test_fetch_paginated_waits_between_chunks():
    page_a = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "epoch": 2000}]
    page_b = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.02, "epoch": 1000}]
    ws = AsyncMock()
    ws.send = AsyncMock(side_effect=[{"candles": page_a}, {"candles": page_b}])
    cfg = parse_history_fetch_config({"history_fetch_chunk": 1, "history_fetch_delay_seconds": 0.01})
    with patch("src.infrastructure.handlers.history_fetch.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        out = await fetch_paginated_candle_history(
            ws,
            symbol="RDBULL",
            granularity=60,
            target=2,
            fetch_cfg=cfg,
            logger=logging.getLogger("test"),
        )
    assert len(out) == 2
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_fetch_paginated_continues_after_partial_first_page():
    partial = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "epoch": 5000 - i} for i in range(999)]
    older = [{"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.02, "epoch": 3000 + i} for i in range(2)]
    ws = AsyncMock()
    ws.send = AsyncMock(side_effect=[{"candles": partial}, {"candles": older}])
    cfg = parse_history_fetch_config({"history_fetch_chunk": 1000, "history_fetch_delay_seconds": 0})
    out = await fetch_paginated_candle_history(
        ws,
        symbol="RDBEAR",
        granularity=60,
        target=1001,
        fetch_cfg=cfg,
        logger=logging.getLogger("test"),
    )
    assert len(out) == 1001
    assert ws.send.await_count == 2
