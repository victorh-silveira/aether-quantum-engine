"""Testes de runtime simulado ate stop win."""

from scripts.backtest.simulator import SettledTrade
from scripts.backtest.simulator_sessions import M15_CANDLE_MINUTES, runtime_to_stop_win
from src.domain.models.trade import TradeDirection


def _trade(bar: int, pnl: float):
    return SettledTrade(
        bar_index=bar,
        symbol="R_25",
        direction=TradeDirection.CALL,
        conviction=0.8,
        macro_tag="risk_on",
        active_region="US",
        return_pct=1.0,
        won=pnl > 0,
        pnl=pnl,
        stake=5.0,
        bankroll_after=100.0,
        session_day=0,
    )


def test_runtime_two_candles_thirty_minutes():
    trades = [_trade(10, 6.0), _trade(11, 5.0)]
    first, stop, minutes, label = runtime_to_stop_win(trades, target=10.0)
    assert first == 10
    assert stop == 11
    assert minutes == 2 * M15_CANDLE_MINUTES
    assert label == "30m"
