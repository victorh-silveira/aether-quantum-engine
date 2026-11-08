import pytest

from src.infrastructure.handlers.tick_buffer import TickBuffer


def test_tick_buffer_aggregates_velocity():
    buf = TickBuffer(["R_10"])
    buf.record_tick("R_10", 1000, 100.0)
    buf.record_tick("R_10", 1100, 100.5)
    buf.record_tick("R_10", 1200, 100.2)
    stats = buf.on_bar_close("R_10", 60)
    assert stats.tick_count == 3.0
    assert stats.mean_inter_tick_ms > 0.0
    series = buf.microstructure_series("R_10", 1)
    assert len(series) == 1
    assert series[0].tick_count == 3.0


def test_tick_buffer_neutral_padding():
    buf = TickBuffer(["R_10"])
    arrays = buf.microstructure_arrays("R_10", 5)
    assert len(arrays["tick_count"]) == 5
    assert arrays["price_velocity"].sum() == 0.0


def test_tick_buffer_ignores_unknown_symbol():
    buf = TickBuffer(["R_10"])
    buf.record_tick("UNKNOWN", 1000, 100.0)
    assert len(buf._live["R_10"]) == 0


def test_tick_buffer_same_epoch_ticks():
    buf = TickBuffer(["R_10"])
    buf.record_tick("R_10", 1000, 100.0)
    buf.record_tick("R_10", 1000, 100.1)
    stats = buf.on_bar_close("R_10", 60)
    assert stats.tick_count == 2.0
    assert stats.mean_inter_tick_ms == 0.0


def test_tick_buffer_two_ticks_only():
    buf = TickBuffer(["R_10"])
    buf.record_tick("R_10", 1000, 100.0)
    buf.record_tick("R_10", 1100, 100.5)
    stats = buf.on_bar_close("R_10", 60)
    assert stats.price_acceleration == 0.0


def test_tick_buffer_live_tick_acceleration_window():
    buf = TickBuffer(["R_10"])
    buf.record_tick("R_10", 55_000, 100.0)
    buf.record_tick("R_10", 56_000, 100.4)
    buf.record_tick("R_10", 57_000, 100.9)
    buf.record_tick("R_10", 58_000, 101.1)
    accel = buf.live_tick_acceleration("R_10", window_ms=5000)
    assert accel != 0.0


def test_tick_buffer_live_tick_acceleration_returns_zero_with_sparse_ticks():
    buf = TickBuffer(["R_10"])
    assert buf.live_tick_acceleration("R_10") == 0.0


def test_tick_buffer_live_tick_acceleration_sparse_window():
    buf = TickBuffer(["R_10"])
    buf.record_tick("R_10", 1000, 100.0)
    buf.record_tick("R_10", 2000, 100.2)
    buf.record_tick("R_10", 30_000, 101.0)
    assert buf.live_tick_acceleration("R_10", window_ms=2000) == 0.0


def test_tick_buffer_live_tick_acceleration_same_epoch_window():
    buf = TickBuffer(["R_10"])
    buf.record_tick("R_10", 5000, 100.0)
    buf.record_tick("R_10", 5000, 100.2)
    buf.record_tick("R_10", 5000, 100.4)
    assert buf.live_tick_acceleration("R_10", window_ms=5000) == 0.0


def test_tick_buffer_reset_live_accumulators_clears_live_ticks():
    buf = TickBuffer(["R_10"])
    buf.record_tick("R_10", 1000, 100.0)
    buf.record_tick("R_10", 1100, 100.5)
    buf.reset_live_accumulators()
    assert len(buf._live["R_10"]) == 0
    assert buf.last_tick_monotonic() == 0.0


@pytest.mark.asyncio
async def test_tick_buffer_last_tick_monotonic_updates_on_record():
    buf = TickBuffer(["R_10"])
    assert buf.last_tick_monotonic() == 0.0
    buf.record_tick("R_10", 1000, 100.0)
    assert buf.last_tick_monotonic() > 0.0
    before = buf.last_tick_monotonic()
    buf.mark_activity()
    assert buf.last_tick_monotonic() >= before


def test_tick_buffer_latest_price_and_epoch():
    buf = TickBuffer(["R_10"])
    assert buf.latest_price("R_10") is None
    assert buf.latest_tick_epoch("R_10") is None
    buf.record_tick("R_10", 1700000000123, 105.5)
    assert buf.latest_price("R_10") == 105.5
    assert buf.latest_tick_epoch("R_10") == pytest.approx(1700000000.123)
    assert buf.latest_tick_epoch("UNKNOWN") is None


def test_tick_buffer_ring_buffer_integration():
    buf = TickBuffer(["1HZ75V"])
    rb = buf.get_ring_buffer("1HZ75V")
    assert rb is not None
    assert rb.count == 0

    buf.record_tick("1HZ75V", 2000, 5000.0)
    assert rb.count == 1
    assert rb.latest_tick() == (2000, 5000.0)

    buf.on_bar_close("1HZ75V", 60)
    assert rb.count == 0
    assert buf.get_ring_buffer("UNKNOWN") is None


def test_tick_buffer_latest_price_and_epoch_corrupted():
    buf = TickBuffer(["R_10"])
    buf._live["R_10"].append(("invalid_epoch", "invalid_price"))
    assert buf.latest_price("R_10") is None
    assert buf.latest_tick_epoch("R_10") is None


def test_tick_buffer_microstructure_expanded_fields():
    buf = TickBuffer(["1HZ75V"])
    buf.record_tick("1HZ75V", 1000, 100.0)
    buf.record_tick("1HZ75V", 2000, 101.0)
    buf.record_tick("1HZ75V", 3000, 100.5)
    buf.record_tick("1HZ75V", 4000, 102.0)
    stats = buf.on_bar_close("1HZ75V", 60)
    assert stats.buy_tick_ratio > 0.0
    assert stats.buy_tick_ratio <= 1.0
    assert stats.realized_volatility > 0.0
    assert stats.final_momentum != 0.0
    assert -1.0 <= stats.wick_rejection_asymmetry <= 1.0
    arrays = buf.microstructure_arrays("1HZ75V", 1)
    assert "buy_tick_ratio" in arrays
    assert "final_momentum" in arrays
    assert "realized_volatility" in arrays
    assert "wick_rejection_asymmetry" in arrays
    assert len(arrays["buy_tick_ratio"]) == 1
