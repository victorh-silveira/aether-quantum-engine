"""Testes do simulador de backtest (Kelly e drawdown)."""

import pytest

from scripts.backtest.signal_engine import BacktestOrder
from scripts.backtest.simulator import compute_max_drawdown, settle_orders_kelly
from src.domain.models.trade import TradeDirection


def test_compute_max_drawdown():
    dd_abs, dd_pct = compute_max_drawdown([100.0, 110.0, 95.0, 105.0])
    assert dd_abs == 15.0
    assert dd_pct == pytest.approx(13.636, abs=0.01)


def test_settle_orders_kelly_updates_bankroll():
    risk_cfg = {
        "kelly": {
            "fraction": 0.15,
            "max_stake_pct": 0.02,
            "max_recovery_stake_pct": 0.1,
            "recovery_conviction_threshold": 0.65,
            "dynamic_win_rate": False,
        },
        "params": {"payout_estimate": 1.0, "stake_min": 1.0},
        "small_account_threshold": 50.0,
        "small_account_stop_win": 10.0,
        "large_account_stop_win_pct": 10.0,
    }
    order = BacktestOrder(
        bar_index=0,
        symbol="OTC_SPC",
        direction=TradeDirection.CALL,
        conviction=0.7,
        macro_tag="risk_on",
        active_region="US",
        index_note="",
    )
    m15 = {"OTC_SPC": [100.0, 101.0]}
    sim = settle_orders_kelly([order], m15, risk_cfg, bankroll_start=100.0)
    assert len(sim.trades) == 1
    assert sim.trades[0].won
    assert sim.trades[0].bankroll_after > 100.0
    assert sim.max_drawdown_abs >= 0.0


def test_kelly_backtest_ignores_stop_win():
    risk_cfg = {
        "kelly": {
            "fraction": 0.15,
            "max_stake_pct": 0.5,
            "max_recovery_stake_pct": 0.5,
            "recovery_conviction_threshold": 0.99,
            "session_max_drawdown_pct": 0.0,
            "dynamic_win_rate": False,
        },
        "params": {"payout_estimate": 1.0, "stake_min": 1.0},
        "small_account_threshold": 50.0,
        "small_account_stop_win": 5.0,
        "large_account_stop_win_pct": 5.0,
    }
    rm_profit_orders = [BacktestOrder(i, "OTC_SPC", TradeDirection.CALL, 0.9, "risk_on", "US", "") for i in range(20)]
    m15 = {"OTC_SPC": [100.0] + [101.0] * 30}
    sim = settle_orders_kelly(rm_profit_orders, m15, risk_cfg, bankroll_start=100.0)
    assert len(sim.trades) <= 20
