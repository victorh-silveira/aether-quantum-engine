"""Testes unitarios para os indicadores matematicos acelerados de fast_indicators."""

import importlib
import math
import sys
from types import ModuleType

import numpy as np
import pytest

from src.domain.math import fast_indicators
from src.domain.math.fast_indicators import (
    _ema_core_python,
    _true_range_core_python,
    fast_atr,
    fast_ema,
    fast_log_returns,
    fast_true_range,
    is_numba_available,
)


def test_is_numba_available_returns_bool():
    res = is_numba_available()
    assert isinstance(res, bool)


def test_fast_ema_empty_and_single():
    assert fast_ema([], span=10).size == 0
    single = fast_ema([42.0], span=10)
    assert len(single) == 1
    assert single[0] == pytest.approx(42.0)


def test_fast_ema_known_values():
    values = [10.0, 20.0, 30.0]
    span = 2
    alpha = 2.0 / (span + 1.0)
    e0 = 10.0
    e1 = alpha * 20.0 + (1.0 - alpha) * e0
    e2 = alpha * 30.0 + (1.0 - alpha) * e1
    res = fast_ema(values, span=span)
    assert len(res) == 3
    assert res[0] == pytest.approx(e0)
    assert res[1] == pytest.approx(e1)
    assert res[2] == pytest.approx(e2)


def test_fast_true_range_empty_and_known():
    assert fast_true_range([], [], []).size == 0

    highs = [105.0, 108.0, 102.0]
    lows = [95.0, 101.0, 98.0]
    closes = [100.0, 102.0, 99.0]

    tr = fast_true_range(highs, lows, closes)
    assert len(tr) == 3
    assert tr[0] == pytest.approx(10.0)
    tr1_expected = max(108.0 - 101.0, abs(108.0 - 100.0), abs(101.0 - 100.0))
    assert tr[1] == pytest.approx(tr1_expected)
    tr2_expected = max(102.0 - 98.0, abs(102.0 - 102.0), abs(98.0 - 102.0))
    assert tr[2] == pytest.approx(tr2_expected)


def test_fast_atr_calculation():
    assert fast_atr([], [], []).size == 0
    highs = [10.0, 12.0, 11.0, 13.0]
    lows = [8.0, 9.0, 9.5, 10.0]
    closes = [9.0, 11.0, 10.0, 12.5]
    atr = fast_atr(highs, lows, closes, period=3)
    assert len(atr) == 4
    assert np.all(atr > 0.0)


def test_fast_log_returns_empty_and_values():
    assert fast_log_returns([]).size == 0
    assert fast_log_returns([100.0]).size == 0

    prices = [100.0, 105.0, 102.0]
    r = fast_log_returns(prices)
    assert len(r) == 2
    assert r[0] == pytest.approx(math.log(105.0 / 100.0))
    assert r[1] == pytest.approx(math.log(102.0 / 105.0))


def test_fast_indicators_cores_empty_arrays():
    empty = np.empty(0, dtype=np.float64)
    assert _ema_core_python(empty, 0.5).size == 0
    assert _true_range_core_python(empty, empty, empty).size == 0


def test_fast_log_returns_with_zero_or_negative_prices():
    prices = [0.0, 100.0, -10.0]
    rets = fast_log_returns(prices)
    assert len(rets) == 2
    assert np.all(np.isfinite(rets))


def test_fast_indicators_reload_with_numba_mock(monkeypatch):
    mock_numba = ModuleType("numba")

    def mock_njit(*_args, **_kwargs):
        def decorator(fn):
            return fn

        return decorator

    mock_numba.njit = mock_njit
    monkeypatch.setitem(sys.modules, "numba", mock_numba)
    importlib.reload(fast_indicators)
    assert fast_indicators.is_numba_available() is True

    monkeypatch.delitem(sys.modules, "numba", raising=False)
    importlib.reload(fast_indicators)
