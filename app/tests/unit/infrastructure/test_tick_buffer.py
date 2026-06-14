from src.infrastructure.handlers.tick_buffer import TickBuffer


def test_tick_buffer_aggregates_velocity():
    buf = TickBuffer(["R_50"])
    buf.record_tick("R_50", 1000, 100.0)
    buf.record_tick("R_50", 1100, 100.5)
    buf.record_tick("R_50", 1200, 100.2)
    stats = buf.on_bar_close("R_50", 60)
    assert stats.tick_count == 3.0
    assert stats.mean_inter_tick_ms > 0.0
    series = buf.microstructure_series("R_50", 1)
    assert len(series) == 1
    assert series[0].tick_count == 3.0


def test_tick_buffer_neutral_padding():
    buf = TickBuffer(["R_50"])
    arrays = buf.microstructure_arrays("R_50", 5)
    assert len(arrays["tick_count"]) == 5
    assert arrays["price_velocity"].sum() == 0.0


def test_tick_buffer_ignores_unknown_symbol():
    buf = TickBuffer(["R_50"])
    buf.record_tick("R_75", 1000, 100.0)
    assert len(buf._live["R_50"]) == 0


def test_tick_buffer_same_epoch_ticks():
    buf = TickBuffer(["R_50"])
    buf.record_tick("R_50", 1000, 100.0)
    buf.record_tick("R_50", 1000, 100.1)
    stats = buf.on_bar_close("R_50", 60)
    assert stats.tick_count == 2.0
    assert stats.mean_inter_tick_ms == 0.0


def test_tick_buffer_two_ticks_only():
    buf = TickBuffer(["R_50"])
    buf.record_tick("R_50", 1000, 100.0)
    buf.record_tick("R_50", 1100, 100.5)
    stats = buf.on_bar_close("R_50", 60)
    assert stats.price_acceleration == 0.0
