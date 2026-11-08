"""Testes para resolucao de timeframe, granularidade e leitura de OHLC."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.application.services.deep_learning.dl_market_data import load_symbol_close_ohlc
from src.application.services.deep_learning.dl_params_timeframe import (
    resolve_dl_granularity,
    resolve_train_timeframe,
)


def test_resolve_train_timeframe_defaults_to_micro():
    assert resolve_train_timeframe(None) == "micro"
    assert resolve_train_timeframe({}) == "micro"
    assert resolve_train_timeframe({"train_timeframe": "micro"}) == "micro"
    assert resolve_train_timeframe({"train_timeframe": "M5"}) == "micro"


def test_resolve_train_timeframe_accepts_macro_and_d1():
    assert resolve_train_timeframe({"train_timeframe": "macro"}) == "macro"
    assert resolve_train_timeframe({"train_timeframe": "d1"}) == "macro"
    assert resolve_train_timeframe({"train_timeframe": "D1"}) == "macro"


def test_resolve_dl_granularity_micro_vs_macro():
    dl_micro = {"train_timeframe": "micro"}
    data_cfg = {"granularity": 86400, "micro_granularity": 300}
    assert resolve_dl_granularity(dl_micro, data_cfg) == 300

    dl_macro = {"train_timeframe": "macro"}
    assert resolve_dl_granularity(dl_macro, data_cfg) == 86400

    dl_empty = {}
    data_empty = {}
    assert resolve_dl_granularity(dl_empty, data_empty) == 180


def test_load_symbol_close_ohlc_defaults_to_micro():
    stream = MagicMock()
    stream.get_micro_numpy_series.side_effect = lambda sym, field: np.array([1.0, 2.0])
    orch = SimpleNamespace(stream=stream)

    close, open_, high, low = load_symbol_close_ohlc(orch, "1HZ75V")
    assert np.array_equal(close, np.array([1.0, 2.0]))
    assert stream.get_micro_numpy_series.called


def test_load_symbol_close_ohlc_explicit_macro():
    stream = MagicMock()
    stream.get_numpy_series.side_effect = lambda sym, field: np.array([10.0, 20.0])
    orch = SimpleNamespace(stream=stream)

    close, open_, high, low = load_symbol_close_ohlc(orch, "1HZ75V", timeframe="macro")
    assert np.array_equal(close, np.array([10.0, 20.0]))
    assert stream.get_numpy_series.called
