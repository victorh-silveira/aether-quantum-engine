"""Testes unitarios para o despachante de ticks colunar."""

import pytest

from src.infrastructure.storage.columnar_tick_store import (
    ColumnarTick,
    ColumnarTickStore,
)


@pytest.mark.asyncio
async def test_columnar_tick_store_push_and_flush():
    store = ColumnarTickStore(batch_size=3, engine_type="questdb_ilp")
    assert store.buffered_count == 0
    assert store.total_flushed == 0

    await store.push_tick("1HZ75V", 1700000000000, 5000.5)
    await store.push_tick("1HZ75V", 1700000001000, 5001.0)
    assert store.buffered_count == 2
    assert store.total_flushed == 0

    await store.push_tick("1HZ75V", 1700000002000, 5001.5)
    assert store.buffered_count == 0
    assert store.total_flushed == 3


@pytest.mark.asyncio
async def test_columnar_tick_store_ilp_formatting():
    store = ColumnarTickStore()
    tick = ColumnarTick(symbol="1HZ75V", epoch_ms=1000, price=100.25)
    line = store.format_ilp_line(tick)

    assert line == "ticks,symbol=1HZ75V price=100.25 1000000000"


@pytest.mark.asyncio
async def test_columnar_tick_store_empty_flush_and_close():
    store = ColumnarTickStore()
    count = await store.flush()
    assert count == 0

    await store.push_tick("1HZ75V", 2000, 200.0)
    assert store.buffered_count == 1
    await store.close()
    assert store.buffered_count == 0
    assert store.total_flushed == 1
