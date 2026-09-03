"""Testes unitarios para execution_anti_loss_helpers com 100% de cobertura."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.application.services.execution_anti_loss_helpers import (
    calc_ema,
    calc_ema_series,
    check_mini_ema_trend_and_slope,
    check_rsi_filter,
)
from src.domain.models.trade import TradeDirection


def test_calc_ema_series_and_calc_ema():
    assert calc_ema_series(np.array([10.0]), 5) is None
    assert calc_ema(np.array([10.0]), 5) is None

    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    s = calc_ema_series(prices, 3)
    assert s is not None
    assert len(s) == len(prices)
    assert calc_ema(prices, 3) == s[-1]


def test_check_mini_ema_trend_and_slope_guards():
    assert check_mini_ema_trend_and_slope(None, "R_10", TradeDirection.CALL) == (True, None)
    assert check_mini_ema_trend_and_slope(MagicMock(), "", TradeDirection.CALL) == (True, None)

    orch = MagicMock()
    orch.stream = None
    assert check_mini_ema_trend_and_slope(orch, "R_10", TradeDirection.CALL) == (True, None)

    stream = MagicMock()
    stream.get_mini_numpy_series.return_value = np.array([10.0, 11.0])
    orch.stream = stream
    assert check_mini_ema_trend_and_slope(orch, "R_10", TradeDirection.CALL) == (True, None)


def test_check_mini_ema_trend_and_slope_full_branches():
    orch = MagicMock()
    stream = MagicMock()
    orch.stream = stream

    # Alta com ATR
    closes_up = np.linspace(100.0, 200.0, 30)
    stream.get_mini_numpy_series.return_value = closes_up
    assert check_mini_ema_trend_and_slope(orch, "R_10", TradeDirection.CALL, metrics={"atr": 2.0}) == (True, None)

    # CALL com preco muito abaixo da EMA9
    closes_call_bad = np.linspace(100.0, 200.0, 30)
    closes_call_bad[-1] = 50.0
    stream.get_mini_numpy_series.return_value = closes_call_bad
    ok, reason = check_mini_ema_trend_and_slope(orch, "R_10", TradeDirection.CALL)
    assert ok is False
    assert reason == "anti_loss_ema_trend"

    # CALL com preco acima da EMA9 mas EMA21 inclinando para baixo
    closes_call_slope = np.linspace(200.0, 100.0, 30)
    closes_call_slope[-1] = 160.0  # acima da EMA9 local mas EMA21 caindo forte
    stream.get_mini_numpy_series.return_value = closes_call_slope
    ok, reason = check_mini_ema_trend_and_slope(orch, "R_10", TradeDirection.CALL)
    assert ok is False
    assert reason == "anti_loss_ema_slope"

    # PUT com preco muito acima da EMA9
    closes_put_bad = np.linspace(200.0, 100.0, 30)
    closes_put_bad[-1] = 250.0
    stream.get_mini_numpy_series.return_value = closes_put_bad
    ok, reason = check_mini_ema_trend_and_slope(orch, "R_10", TradeDirection.PUT)
    assert ok is False
    assert reason == "anti_loss_ema_trend"

    # PUT com preco abaixo da EMA9 mas EMA21 subindo forte
    closes_put_slope = np.linspace(100.0, 200.0, 30)
    closes_put_slope[-1] = 140.0  # abaixo da EMA9 local mas EMA21 subindo forte
    stream.get_mini_numpy_series.return_value = closes_put_slope
    ok, reason = check_mini_ema_trend_and_slope(orch, "R_10", TradeDirection.PUT)
    assert ok is False
    assert reason == "anti_loss_ema_slope"

    # PUT em tendencia de baixa normal
    stream.get_mini_numpy_series.return_value = np.linspace(200.0, 100.0, 30)
    assert check_mini_ema_trend_and_slope(orch, "R_10", TradeDirection.PUT) == (True, None)


def test_check_rsi_filter():
    assert check_rsi_filter({}, TradeDirection.CALL) is True
    assert check_rsi_filter({"indicators": {"rsi": "invalid"}}, TradeDirection.CALL) is True
    assert check_rsi_filter({"indicators": {"rsi": 0.30}}, TradeDirection.CALL) is False
    assert check_rsi_filter({"indicators": {"rsi": 0.50}}, TradeDirection.CALL) is True
    assert check_rsi_filter({"indicators": {"rsi": 70.0}}, TradeDirection.PUT) is False
    assert check_rsi_filter({"indicators": {"rsi": 50.0}}, TradeDirection.PUT) is True
    assert check_rsi_filter({"micro_indicators": {"rsi": 0.25}}, TradeDirection.CALL) is False
    # Limites customizados
    assert check_rsi_filter({"indicators": {"rsi": 0.36}}, TradeDirection.CALL, rsi_min=0.38) is False
    assert check_rsi_filter({"indicators": {"rsi": 0.40}}, TradeDirection.CALL, rsi_min=0.38) is True
    assert check_rsi_filter({"indicators": {"rsi": 0.63}}, TradeDirection.PUT, rsi_max=0.62) is False
    assert check_rsi_filter({"indicators": {"rsi": 0.60}}, TradeDirection.PUT, rsi_max=0.62) is True
