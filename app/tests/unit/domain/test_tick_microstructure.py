"""Testes unitarios para compute_tick_microstructure e compute_buffer_microstructure."""

from datetime import UTC, datetime

import polars as pl

from src.application.services.tick_microstructure_frame import compute_tick_microstructure
from src.domain.analytics.tick_microstructure import compute_buffer_microstructure


def test_compute_tick_microstructure_empty():
    df_empty = pl.DataFrame(schema={"timestamp": pl.Datetime("ms"), "price": pl.Float64})
    result = compute_tick_microstructure(df_empty)
    assert result.is_empty()
    assert "buy_tick_ratio" in result.columns
    assert "return_autocorr" in result.columns
    assert "final_momentum" in result.columns
    assert "realized_volatility" in result.columns


def test_compute_tick_microstructure_numeric_timestamp():
    base_time = 1700000100
    prices = [100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0]
    timestamps = [base_time + i * 30 for i in range(len(prices))]

    df = pl.DataFrame({"timestamp": timestamps, "price": prices})
    result = compute_tick_microstructure(df, window_seconds=300)

    assert not result.is_empty()
    assert "buy_tick_ratio" in result.columns
    row = result.to_dicts()[0]
    assert row["open"] == 100.0
    assert row["high"] == 104.0
    assert row["low"] == 100.0
    assert row["close"] == 104.0
    assert 0.0 <= row["buy_tick_ratio"] <= 1.0
    assert isinstance(row["realized_volatility"], float)
    assert row["realized_volatility"] >= 0.0


def test_compute_tick_microstructure_datetime_timestamp():
    dt_base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    timestamps = [datetime.fromtimestamp(dt_base.timestamp() + i * 10, tz=UTC) for i in range(10)]
    prices = [10.0 + i * 0.5 for i in range(10)]

    df = pl.DataFrame({"timestamp": timestamps, "price": prices})
    result = compute_tick_microstructure(df, window_seconds=300)

    assert not result.is_empty()
    row = result.to_dicts()[0]
    assert row["buy_tick_ratio"] == 1.0
    assert row["final_momentum"] > 0.0


def test_compute_buffer_microstructure_empty_and_short():
    fallback = compute_buffer_microstructure([])
    assert fallback["buy_tick_ratio"] == 0.5
    assert fallback["return_autocorr"] == 0.0
    assert fallback["final_momentum"] == 0.0
    assert fallback["realized_volatility"] == 0.0

    single = compute_buffer_microstructure([(1.0, 100.0)])
    assert single["buy_tick_ratio"] == 0.5
    pair = compute_buffer_microstructure([(1.0, 100.0), (2.0, 101.0)])
    assert pair["buy_tick_ratio"] == 1.0
    assert pair["return_autocorr"] == 0.0


def test_compute_buffer_microstructure_valid_ticks():
    ticks = [(float(i), 100.0 + float(i)) for i in range(50)]
    metrics = compute_buffer_microstructure(ticks, window_seconds=300)

    assert metrics["buy_tick_ratio"] == 1.0
    assert metrics["final_momentum"] > 0.0
    assert metrics["realized_volatility"] > 0.0
    assert isinstance(metrics["return_autocorr"], float)


def test_compute_buffer_microstructure_downward_and_short_tail():
    ticks = [(float(i), 100.0 - float(i)) for i in range(10)]
    metrics = compute_buffer_microstructure(ticks, window_seconds=300)

    assert metrics["buy_tick_ratio"] == 0.0
    assert metrics["final_momentum"] < 0.0
    assert metrics["realized_volatility"] > 0.0
