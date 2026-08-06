import pytest

from src.infrastructure.handlers.tick_buffer import TickBuffer


def test_tick_buffer_aggregates_velocity():
    buf = TickBuffer(["OTC_SPC"])
    buf.record_tick("OTC_SPC", 1000, 100.0)
    buf.record_tick("OTC_SPC", 1100, 100.5)
    buf.record_tick("OTC_SPC", 1200, 100.2)
    stats = buf.on_bar_close("OTC_SPC", 60)
    assert stats.tick_count == 3.0
    assert stats.mean_inter_tick_ms > 0.0
    series = buf.microstructure_series("OTC_SPC", 1)
    assert len(series) == 1
    assert series[0].tick_count == 3.0


def test_tick_buffer_neutral_padding():
    buf = TickBuffer(["OTC_SPC"])
    arrays = buf.microstructure_arrays("OTC_SPC", 5)
    assert len(arrays["tick_count"]) == 5
    assert arrays["price_velocity"].sum() == 0.0


def test_tick_buffer_ignores_unknown_symbol():
    buf = TickBuffer(["OTC_SPC"])
    buf.record_tick("UNKNOWN", 1000, 100.0)
    assert len(buf._live["OTC_SPC"]) == 0


def test_tick_buffer_same_epoch_ticks():
    buf = TickBuffer(["OTC_SPC"])
    buf.record_tick("OTC_SPC", 1000, 100.0)
    buf.record_tick("OTC_SPC", 1000, 100.1)
    stats = buf.on_bar_close("OTC_SPC", 60)
    assert stats.tick_count == 2.0
    assert stats.mean_inter_tick_ms == 0.0


def test_tick_buffer_two_ticks_only():
    buf = TickBuffer(["OTC_SPC"])
    buf.record_tick("OTC_SPC", 1000, 100.0)
    buf.record_tick("OTC_SPC", 1100, 100.5)
    stats = buf.on_bar_close("OTC_SPC", 60)
    assert stats.price_acceleration == 0.0


def test_tick_buffer_live_tick_acceleration_window():
    buf = TickBuffer(["OTC_SPC"])
    buf.record_tick("OTC_SPC", 55_000, 100.0)
    buf.record_tick("OTC_SPC", 56_000, 100.4)
    buf.record_tick("OTC_SPC", 57_000, 100.9)
    buf.record_tick("OTC_SPC", 58_000, 101.1)
    accel = buf.live_tick_acceleration("OTC_SPC", window_ms=5000)
    assert accel != 0.0


def test_tick_buffer_live_tick_acceleration_returns_zero_with_sparse_ticks():
    buf = TickBuffer(["OTC_SPC"])
    assert buf.live_tick_acceleration("OTC_SPC") == 0.0


def test_tick_buffer_live_tick_acceleration_sparse_window():
    buf = TickBuffer(["OTC_SPC"])
    buf.record_tick("OTC_SPC", 1000, 100.0)
    buf.record_tick("OTC_SPC", 2000, 100.2)
    buf.record_tick("OTC_SPC", 30_000, 101.0)
    assert buf.live_tick_acceleration("OTC_SPC", window_ms=2000) == 0.0


def test_tick_buffer_live_tick_acceleration_same_epoch_window():
    buf = TickBuffer(["OTC_SPC"])
    buf.record_tick("OTC_SPC", 5000, 100.0)
    buf.record_tick("OTC_SPC", 5000, 100.2)
    buf.record_tick("OTC_SPC", 5000, 100.4)
    assert buf.live_tick_acceleration("OTC_SPC", window_ms=5000) == 0.0


def test_tick_buffer_reset_live_accumulators_clears_live_ticks():
    buf = TickBuffer(["OTC_SPC"])
    buf.record_tick("OTC_SPC", 1000, 100.0)
    buf.record_tick("OTC_SPC", 1100, 100.5)
    buf.reset_live_accumulators()
    assert len(buf._live["OTC_SPC"]) == 0
    assert buf.last_tick_monotonic() == 0.0


@pytest.mark.asyncio
async def test_tick_buffer_last_tick_monotonic_updates_on_record():
    buf = TickBuffer(["OTC_SPC"])
    assert buf.last_tick_monotonic() == 0.0
    buf.record_tick("OTC_SPC", 1000, 100.0)
    assert buf.last_tick_monotonic() > 0.0
    before = buf.last_tick_monotonic()
    buf.touch_activity()
    assert buf.last_tick_monotonic() >= before
