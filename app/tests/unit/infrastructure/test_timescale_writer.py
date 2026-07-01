"""Testes do TimescaleMarketWriter com pool mockado."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.market.null_market_writer import NullMarketWriter
from src.infrastructure.market.timescale_writer import TimescaleMarketWriter


@pytest.mark.asyncio
async def test_null_market_writer_noop():
    writer = NullMarketWriter()
    await writer.enqueue_tick(symbol="R_10", epoch_ms=1, price=1.0)
    await writer.enqueue_bar(symbol="R_10", bar={"epoch": 1})
    assert await writer.ping() is True


@pytest.mark.asyncio
async def test_timescale_writer_enqueue_and_flush():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db", flush_interval_ms=1.0, batch_limit=2)
    conn = AsyncMock()
    conn.executemany = AsyncMock()
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    with patch.object(writer, "_ensure_pool", AsyncMock(return_value=pool)):
        await writer.enqueue_tick(symbol="R_10", epoch_ms=1000, price=1.23)
        await writer.enqueue_bar(
            symbol="R_10",
            bar={
                "epoch": 1,
                "granularity": 300,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "tick_count": 3,
                "mean_inter_tick_ms": 10.0,
                "price_velocity": 0.1,
            },
        )
        await writer.flush()
    assert conn.executemany.await_count >= 1


@pytest.mark.asyncio
async def test_timescale_ping():
    writer = TimescaleMarketWriter(dsn="postgresql://u:p@localhost/db")
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=1)
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    with patch.object(writer, "_ensure_pool", AsyncMock(return_value=pool)):
        assert await writer.ping() is True
