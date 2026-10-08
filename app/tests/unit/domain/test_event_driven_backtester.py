"""Testes unitarios para o backtester orientado a eventos com especificacoes Deriv."""

import numpy as np
import pytest

from src.domain.analytics.event_driven_backtester import (
    EventDrivenBacktester,
    TradeSignal,
)


def test_event_driven_backtester_empty_or_out_of_bounds():
    backtester = EventDrivenBacktester(horizon_steps=5, execution_delay_steps=1)
    report = backtester.run(np.array([100.0, 101.0]), [])
    assert report.total_trades == 0
    assert report.total_pnl == 0.0

    signals = [TradeSignal(index=10, direction="CALL", stake=10.0)]
    report2 = backtester.run(np.array([100.0, 101.0, 102.0]), signals)
    assert report2.total_trades == 0


def test_event_driven_backtester_call_and_put_wins():
    prices = np.array([100.0, 101.0, 105.0, 103.0, 99.0], dtype=np.float64)
    backtester = EventDrivenBacktester(horizon_steps=1, execution_delay_steps=0, payout_rate=0.85)

    signals = [
        TradeSignal(index=1, direction="CALL", stake=100.0),
        TradeSignal(index=3, direction="PUT", stake=100.0),
    ]

    report = backtester.run(prices, signals)
    assert report.total_trades == 2
    assert report.wins == 2
    assert report.losses == 0
    assert report.win_rate == 1.0
    assert report.total_pnl == pytest.approx(170.0)
    assert report.profit_factor == float("inf")


def test_event_driven_backtester_tie_results_in_loss():
    prices = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    backtester = EventDrivenBacktester(horizon_steps=1, execution_delay_steps=0, payout_rate=0.85)

    signals = [
        TradeSignal(index=0, direction="CALL", stake=50.0),
        TradeSignal(index=1, direction="PUT", stake=50.0),
    ]

    report = backtester.run(prices, signals)
    assert report.total_trades == 2
    assert report.wins == 0
    assert report.losses == 2
    assert report.win_rate == 0.0
    assert report.total_pnl == -100.0


def test_event_driven_backtester_execution_delay_slippage():
    prices = np.array([100.0, 105.0, 103.0], dtype=np.float64)
    backtester_no_delay = EventDrivenBacktester(horizon_steps=1, execution_delay_steps=0)
    sig = [TradeSignal(index=0, direction="CALL", stake=10.0)]
    report_no_delay = backtester_no_delay.run(prices, sig)
    assert report_no_delay.wins == 1

    backtester_with_delay = EventDrivenBacktester(horizon_steps=1, execution_delay_steps=1)
    report_with_delay = backtester_with_delay.run(prices, sig)
    assert report_with_delay.losses == 1


def test_event_driven_backtester_drawdown_and_sharpe():
    prices = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 105.0], dtype=np.float64)
    backtester = EventDrivenBacktester(horizon_steps=1, execution_delay_steps=0, payout_rate=0.85)

    signals = [
        TradeSignal(index=0, direction="CALL", stake=100.0),
        TradeSignal(index=1, direction="CALL", stake=100.0),
        TradeSignal(index=2, direction="CALL", stake=100.0),
    ]

    report = backtester.run(prices, signals)
    assert report.total_trades == 3
    assert report.wins == 2
    assert report.losses == 1
    assert report.max_drawdown > 0.0
    assert report.profit_factor > 1.0
    assert isinstance(report.sharpe_ratio, float)
