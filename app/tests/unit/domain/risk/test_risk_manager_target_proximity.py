import pytest

from src.domain.risk.risk_manager import RiskManager


def test_apply_kelly_target_proximity_damping_via_manager(kelly_config):
    rm = RiskManager(kelly_config)
    rm.initial_bankroll = 1000.0
    rm.total_session_profit = 0.0
    target = 10.0
    assert rm.apply_kelly_target_proximity_damping(20.0, target_win=target) == pytest.approx(20.0)
    rm.total_session_profit = 9.0
    assert rm.apply_kelly_target_proximity_damping(20.0, target_win=target) == pytest.approx(20.0 * 0.46)


def test_apply_kelly_target_proximity_damping_resolves_target_from_config(kelly_config):
    rm = RiskManager({**kelly_config, "params": {**kelly_config["params"], "compounding_enabled": True}})
    rm.initial_bankroll = 1000.0
    rm.daily_stop_win_target = 10.0
    rm.total_session_profit = 0.0
    assert rm.apply_kelly_target_proximity_damping(15.0) == pytest.approx(15.0)
