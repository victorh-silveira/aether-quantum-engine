"""Contratos de dominio para catalogo e velas."""

from indicator.domain import Candle, aggregate, closed_candles, latest_contiguous, synthetic_symbols


def test_synthetic_catalog_filters_forex_suspended_and_bad_symbol():
    rows = [
        {"market": "synthetic_index", "underlying_symbol": "R_75", "underlying_symbol_name": "Volatility 75"},
        {"market": "forex", "underlying_symbol": "EURUSD"},
        {"market": "synthetic_index", "underlying_symbol": 'BAD"LABEL'},
        {"market": "synthetic_index", "underlying_symbol": "STOP", "is_trading_suspended": 1},
        {"market": "synthetic_index", "symbol": "1HZ75V", "display_name": "V75"},
    ]
    assert synthetic_symbols(rows) == {"1HZ75V": "V75", "R_75": "Volatility 75"}


def test_closed_candles_rejects_open_invalid_and_duplicates():
    rows = [
        {"epoch": 300, "open": 2, "high": 3, "low": 1, "close": 2},
        {"epoch": 300, "open": 2, "high": 4, "low": 1, "close": 3},
        {"epoch": 600, "open": 2, "high": 3, "low": 1, "close": 2},
        {"epoch": 301, "open": 2, "high": 3, "low": 1, "close": 2},
        {"epoch": 0, "open": "nan", "high": 3, "low": 1, "close": 2},
        {"epoch": 0, "open": 2},
    ]
    assert closed_candles(rows, 900) == [Candle(300, 2, 4, 1, 3), Candle(600, 2, 3, 1, 2)]
    assert closed_candles(rows, 700) == [Candle(300, 2, 4, 1, 3)]


def test_aggregate_requires_complete_uninterpolated_group():
    candles = [Candle(i * 300, i + 1, i + 2, i, i + 1) for i in range(12)]
    assert aggregate(candles, 300) == candles
    assert aggregate(candles, 900)[0] == Candle(0, 1, 4, 0, 3)
    assert len(aggregate(candles, 3600)) == 1
    assert aggregate(candles[1:], 3600) == []
    assert aggregate(candles[:1] + candles[2:3], 900) == []
    try:
        aggregate(candles, 60)
    except ValueError:
        pass
    else:
        raise AssertionError("periodo nao suportado")


def test_latest_contiguous_keeps_recent_segment():
    candles = [Candle(epoch, 1, 1, 1, 1) for epoch in (0, 300, 900, 1200)]
    assert [c.epoch for c in latest_contiguous(candles, 300)] == [900, 1200]
    assert latest_contiguous([], 300) == []
