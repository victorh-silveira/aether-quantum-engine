"""Contratos de dominio para catalogo e velas."""

from indicator.domain import (
    Candle,
    Signal,
    aggregate,
    closed_candles,
    current_spot,
    latest_contiguous,
    live_observation,
    synthetic_symbols,
)


def test_synthetic_catalog_filters_forex_suspended_and_bad_symbol():
    rows = [
        {"market": "synthetic_index", "underlying_symbol": "R_75", "underlying_symbol_name": "Volatility 75"},
        {"market": "forex", "underlying_symbol": "EURUSD"},
        {"market": "synthetic_index", "underlying_symbol": 'BAD"LABEL'},
        {"market": "synthetic_index", "underlying_symbol": "STOP", "is_trading_suspended": 1},
        {"market": "synthetic_index", "symbol": "1HZ75V", "display_name": "V75"},
        {"market": "synthetic_index", "symbol": "R_10", "display_name": "Volatility 10"},
        {"market": "synthetic_index", "symbol": "1HZ25V", "display_name": "Volatility 25 (1s)"},
        {"market": "synthetic_index", "symbol": "BOOM1000", "display_name": "Boom"},
        {"market": "synthetic_index", "symbol": "R_ABC", "display_name": "Bad"},
        {"market": "synthetic_index", "symbol": "RDBEAR", "display_name": "Bear Market Index"},
        {"market": "synthetic_index", "symbol": "RDBULL", "display_name": "Bull Market Index"},
    ]
    assert synthetic_symbols(rows) == {
        "1HZ25V": "Volatility 25 (1s)",
        "1HZ75V": "V75",
        "RDBEAR": "Bear Market Index",
        "RDBULL": "Bull Market Index",
        "R_10": "Volatility 10",
        "R_75": "Volatility 75",
    }


def test_closed_candles_rejects_open_invalid_and_duplicates():
    rows = [
        {"epoch": 300, "open": 2, "high": 3, "low": 1, "close": 2},
        {"epoch": 300, "open": 2, "high": 4, "low": 1, "close": 3},
        {"epoch": 600, "open": 2, "high": 3, "low": 1, "close": 2},
        {"epoch": 301, "open": 2, "high": 3, "low": 1, "close": 2},
        {"epoch": 0, "open": "nan", "high": 3, "low": 1, "close": 2},
        {"epoch": 0, "open": 2},
    ]
    assert closed_candles(rows, 900, period=300) == [Candle(600, 2, 3, 1, 2)]
    assert closed_candles(rows, 700, period=300) == []
    m15_rows = [{"epoch": 900, "open": 10, "high": 12, "low": 9, "close": 11}]
    assert closed_candles(m15_rows, 1800) == [Candle(900, 10, 12, 9, 11)]


def test_aggregate_requires_complete_uninterpolated_group():
    assert aggregate([], 900) == []
    candles = [Candle(i * 300, i + 1, i + 2, i, i + 1) for i in range(12)]
    assert aggregate(candles, 300) == candles
    assert aggregate(candles, 900)[0] == Candle(0, 1, 4, 0, 3)
    assert len(aggregate(candles, 3600)) == 1
    assert aggregate(candles[1:], 3600) == []
    assert aggregate(candles[:1] + candles[2:3], 900) == []
    m15_candles = [Candle(i * 900, i + 1, i + 2, i, i + 1) for i in range(96)]
    assert aggregate(m15_candles, 900) == m15_candles
    assert len(aggregate(m15_candles, 86400)) == 1
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


def test_live_price_and_provisional_outcome_are_causal():
    rows = [{"epoch": 600, "close": 100}, {"epoch": 900, "close": 101}]
    assert current_spot(rows, 960, period=300) == 101
    assert current_spot(rows, 960) == 101
    assert current_spot(rows, 1800) is None
    assert current_spot([{"epoch": 900, "close": "nan"}], 960) is None
    assert current_spot([{"epoch": 900}], 960) is None
    signal = Signal("R_75", 300, 600, "PUT", 0.4, "v1", "ok", 100)
    live = live_observation(signal, 101, 960)
    assert live is not None and live.interim_side == "CALL" and live.delta == 1
    assert live_observation(signal, 99, 960).interim_side == "PUT"
    assert live_observation(signal, 100, 960).interim_side == "EMPATE"
    assert live_observation(signal, 101, 900) is not None
    assert live_observation(signal, 101, 899) is None
    assert live_observation(signal, 101, 1200) is None
    assert live_observation(Signal("R_75", 300, 600, "SEM SINAL", None, None, "sem_modelo", 100), 101, 960) is None
