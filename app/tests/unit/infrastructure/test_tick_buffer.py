import pytest

from src.infrastructure.handlers.tick_buffer import TickBuffer


def test_tick_buffer_aggregates_velocity():
    buf = TickBuffer(["RDBULL"])
    buf.record_tick("RDBULL", 1000, 100.0)
    buf.record_tick("RDBULL", 1100, 100.5)
    buf.record_tick("RDBULL", 1200, 100.2)
    stats = buf.on_bar_close("RDBULL", 60)
    assert stats.tick_count == 3.0
    assert stats.mean_inter_tick_ms > 0.0
    series = buf.microstructure_series("RDBULL", 1)
    assert len(series) == 1
    assert series[0].tick_count == 3.0


def test_tick_buffer_neutral_padding():
    buf = TickBuffer(["RDBULL"])
    arrays = buf.microstructure_arrays("RDBULL", 5)
    assert len(arrays["tick_count"]) == 5
    assert arrays["price_velocity"].sum() == 0.0


def test_tick_buffer_ignores_unknown_symbol():
    buf = TickBuffer(["RDBULL"])
    buf.record_tick("RDBEAR", 1000, 100.0)
    assert len(buf._live["RDBULL"]) == 0


def test_tick_buffer_same_epoch_ticks():
    buf = TickBuffer(["RDBULL"])
    buf.record_tick("RDBULL", 1000, 100.0)
    buf.record_tick("RDBULL", 1000, 100.1)
    stats = buf.on_bar_close("RDBULL", 60)
    assert stats.tick_count == 2.0
    assert stats.mean_inter_tick_ms == 0.0


def test_tick_buffer_two_ticks_only():
    buf = TickBuffer(["RDBULL"])
    buf.record_tick("RDBULL", 1000, 100.0)
    buf.record_tick("RDBULL", 1100, 100.5)
    stats = buf.on_bar_close("RDBULL", 60)
    assert stats.price_acceleration == 0.0


@pytest.mark.asyncio
async def test_tick_buffer_last_tick_monotonic_updates_on_record():
    buf = TickBuffer(["RDBULL"])
    assert buf.last_tick_monotonic() == 0.0
    buf.record_tick("RDBULL", 1000, 100.0)
    assert buf.last_tick_monotonic() > 0.0
    before = buf.last_tick_monotonic()
    buf.touch_activity()
    assert buf.last_tick_monotonic() >= before
