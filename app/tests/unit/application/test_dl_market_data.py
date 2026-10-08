from types import SimpleNamespace

import numpy as np

from src.application.services.deep_learning.dl_market_data import (
    closed_model_ohlc,
    load_symbol_close_ohlc,
    slice_ohlc_window,
)


class _Stream:
    def __init__(self, series: dict[str, np.ndarray]):
        self._series = series

    def get_numpy_series(self, _symbol: str, field: str) -> np.ndarray:
        return self._series[field]


def test_load_symbol_close_ohlc_empty():
    orch = type("O", (), {"stream": _Stream({"close": np.array([])})})()
    close, open_, high, low = load_symbol_close_ohlc(orch, "R_10")
    assert len(close) == 0
    assert open_ is None


def test_load_symbol_close_ohlc_mismatched_lengths():
    close = np.ones(8)
    orch = type(
        "O",
        (),
        {
            "stream": _Stream(
                {
                    "close": close,
                    "open": np.ones(5),
                    "high": close,
                    "low": close,
                }
            )
        },
    )()
    out_close, open_, high, low = load_symbol_close_ohlc(orch, "R_10")
    assert len(out_close) == 8
    assert open_ is None and high is None and low is None


def test_load_symbol_close_ohlc_aligned():
    close = np.arange(6.0)
    orch = type(
        "O",
        (),
        {
            "stream": _Stream(
                {
                    "close": close,
                    "open": close + 0.1,
                    "high": close + 0.2,
                    "low": close - 0.1,
                }
            )
        },
    )()
    c, o, h, low = load_symbol_close_ohlc(orch, "R_10")
    assert len(c) == 6 and o is not None and h is not None and low is not None


def test_load_symbol_close_ohlc_micro_timeframe():
    close = np.arange(5.0)

    class _MicroStream:
        def get_numpy_series(self, _symbol: str, _field: str) -> np.ndarray:
            raise AssertionError("macro path should not be used")

        def get_micro_numpy_series(self, _symbol: str, field: str) -> np.ndarray:
            base = {"close": close, "open": close + 0.1, "high": close + 0.2, "low": close - 0.1}
            return base[field]

    orch = type("O", (), {"stream": _MicroStream()})()
    c, o, h, low = load_symbol_close_ohlc(orch, "R_10", timeframe="micro")
    assert len(c) == 5 and o is not None and h is not None and low is not None


def test_load_symbol_close_ohlc_micro_empty():
    class _MicroStream:
        def get_micro_numpy_series(self, _symbol: str, _field: str) -> np.ndarray:
            return np.array([])

    orch = type("O", (), {"stream": _MicroStream()})()
    close, open_, high, low = load_symbol_close_ohlc(orch, "R_10", timeframe="micro")
    assert len(close) == 0
    assert open_ is None and high is None and low is None


def test_slice_ohlc_window_with_full_ohlc():
    close = np.arange(10.0)
    open_ = close + 1.0
    high = close + 2.0
    low = close - 1.0
    tc, to, th, tl = slice_ohlc_window(close, open_, high, low, start=3)
    assert len(tc) == 7
    assert to[0] == 4.0 and th[-1] == 11.0 and tl[0] == 2.0


def test_slice_ohlc_window_close_only():
    close = np.arange(5.0)
    tc, to, th, tl = slice_ohlc_window(close, None, None, None, start=2)
    assert len(tc) == 3 and to is None


def test_closed_model_ohlc_excludes_only_forming_contract_bar():
    close = np.array([10.0, 11.0, 12.0])
    orch = SimpleNamespace(
        stream=SimpleNamespace(
            micro_candles={
                "1HZ75V": [SimpleNamespace(epoch=600), SimpleNamespace(epoch=900), SimpleNamespace(epoch=1200)]
            }
        )
    )
    result = closed_model_ohlc(orch, "1HZ75V", close, close, close, close, granularity=300, now_epoch=1212)
    assert all(np.array_equal(part, close[:2]) for part in result)
    closed = closed_model_ohlc(orch, "1HZ75V", close, close, close, close, granularity=300, now_epoch=1500)
    assert all(np.array_equal(part, close) for part in closed)


def test_closed_model_ohlc_keeps_series_without_aligned_candle_history():
    close = np.array([10.0, 11.0, 12.0])
    orch = SimpleNamespace(stream=SimpleNamespace(micro_candles={"1HZ75V": [SimpleNamespace(epoch=1200)]}))
    assert closed_model_ohlc(orch, "1HZ75V", close, None, None, None, granularity=300, now_epoch=1212)[0] is close
    orch.stream.micro_candles["1HZ75V"].append(SimpleNamespace(epoch=1500))
    assert len(closed_model_ohlc(orch, "1HZ75V", close, None, None, None, granularity=300, now_epoch=1512)[0]) == 0
