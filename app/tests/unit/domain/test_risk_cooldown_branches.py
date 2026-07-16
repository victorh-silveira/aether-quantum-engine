"""Testes de cobertura para RiskManager Kelly."""

import pytest

from src.domain.risk.risk_manager import RiskManager


def test_risk_manager_rolling_wins_cap():
    """Cobre o limite de amostras no histórico de vitórias."""
    rm = RiskManager({})
    for _ in range(110):
        rm.record_trade_outcome("SYM", won=True)
    wr, n = rm.get_wr_rolling_stats("SYM")
    assert n == 100


def test_risk_manager_cooldown_always_inactive():
    """Garante que o cooldown de ticks permanece inativo."""
    rm = RiskManager({"params": {"entry_cooldown_ticks": 10}})
    rm.last_result_tick = 100
    assert rm.is_on_cooldown(105) is False
    assert rm.is_on_cooldown(115) is False


def test_risk_manager_high_conviction_no_cooldown():
    rm = RiskManager(
        {
            "params": {
                "entry_cooldown_ticks": 12,
                "entry_cooldown_ticks_high_conviction": 6,
                "high_conviction_cooldown_threshold": 0.85,
            }
        }
    )
    rm.current_cooldown_ticks = 12
    rm.register_entry_conviction(0.9)
    rm.last_result_tick = 100
    assert rm.effective_cooldown_ticks() == 0
    assert rm.is_on_cooldown(105) is False
    assert rm.is_on_cooldown(107) is False


def test_risk_manager_effective_cooldown_when_active_zero():
    rm = RiskManager(
        {
            "params": {
                "entry_cooldown_ticks": 12,
                "entry_cooldown_ticks_high_conviction": 6,
                "high_conviction_cooldown_threshold": 0.85,
            }
        }
    )
    rm.current_cooldown_ticks = 0
    rm.register_entry_conviction(0.9)
    assert rm.effective_cooldown_ticks() == 0


def test_risk_manager_effective_cooldown_target_zero():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 0}})
    rm.current_cooldown_ticks = 8
    assert rm.effective_cooldown_ticks() == 0


def test_risk_manager_seconds_cooldown_inactive():
    rm = RiskManager({"params": {"entry_cooldown_seconds": 30}})
    rm.register_entry_conviction(0.7)
    rm._arm_cooldown_timer()
    assert rm._uses_seconds_cooldown() is False
    assert rm.is_on_cooldown(1) is False


def test_cooldown_span_seconds_always_zero():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 3}})
    rm.set_candle_interval_seconds(120)
    rm.current_cooldown_ticks = 3
    assert rm._cooldown_span_seconds() == 0.0


def test_is_on_cooldown_always_false_with_mono_timer():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 5}})
    rm._cooldown_until_mono = 120.0
    assert rm.is_on_cooldown(0) is False


def test_cooldown_span_with_override_always_zero():
    rm = RiskManager(
        {
            "params": {
                "entry_cooldown_ticks": 4,
                "entry_cooldown_tick_seconds": 15,
            }
        }
    )
    rm.current_cooldown_ticks = 4
    assert rm._cooldown_span_seconds() == 0.0


def test_risk_manager_no_cooldown_after_cluster_finalize():
    rm = RiskManager(
        {
            "params": {"entry_cooldown_ticks": 8},
            "kelly": {},
        }
    )
    rm.set_candle_interval_seconds(900)
    rm.current_cooldown_ticks = 16
    rm.register_entry_conviction(0.7)
    rm.last_result_tick = 10
    rm.cluster_results = {1: -1.0}
    rm.expected_cluster_settlements = 1
    rm._finalize_cluster()
    assert rm.is_on_cooldown(10) is False
    assert rm.cooldown_remaining_seconds() == 0.0


def test_stake_block_reason_stop_win():
    rm = RiskManager({"small_account_stop_win": 5.0, "small_account_threshold": 100.0, "params": {"stake_min": 1.0}})
    rm.set_initial_bankroll(50.0)
    rm.total_session_profit = 10.0
    assert rm.stake_block_reason(50.0, "RDBULL") == "stop_win"


def test_arm_cooldown_timer_no_op():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 0}})
    rm._cooldown_until_mono = 999.0
    rm._arm_cooldown_timer()
    assert rm._cooldown_until_mono == 999.0
    assert rm.is_on_cooldown(99) is False


def test_stake_block_reason_stop_win_with_persisted_target():
    rm = RiskManager(
        {
            "kelly": {"fraction": 0.1},
            "params": {
                "compounding_enabled": True,
                "compounding_rate_daily": 0.026,
                "payout_estimate": 0.95,
                "stake_min": 1.0,
            },
        }
    )
    rm.set_initial_bankroll(1000.0)
    rm.daily_stop_win_target = 10.0
    rm.total_session_profit = 11.0
    assert rm.stake_block_reason(1000.0, "RDBULL", conviction=0.6) == "stop_win"


def test_risk_manager_reset_session():
    rm = RiskManager({"params": {}, "kelly": {}, "dlambert": {}})
    rm.reset_session(500.0, target=5.0)
    assert rm.initial_bankroll == 500.0
    assert rm.daily_stop_win_target == 5.0


def test_stake_block_reason_kelly_no_edge():
    rm = RiskManager(
        {
            "kelly": {"fraction": 0.01},
            "params": {"payout_estimate": 0.5, "stake_min": 1.0},
        }
    )
    rm.set_initial_bankroll(1000.0)
    assert rm.stake_block_reason(1000.0, "RDBULL", conviction=0.05) is None
    assert rm.calculate_stake(1000.0, "RDBULL", conviction=0.05) == pytest.approx(1.50)


def test_stake_block_reason_kelly_no_edge_when_bankroll_below_stake_min():
    rm = RiskManager(
        {
            "kelly": {"fraction": 0.01},
            "params": {"payout_estimate": 0.5, "stake_min": 1.0},
        }
    )
    rm.set_initial_bankroll(0.5)
    assert rm.stake_block_reason(0.5, "RDBULL", conviction=0.05) == "kelly_no_edge"


def test_cooldown_mono_expiry_fallback_no_op():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 10}})
    rm.set_candle_interval_seconds(60)
    rm._cooldown_until_mono = 0.0
    rm.last_result_tick = 100
    rm.current_cooldown_ticks = 10
    assert rm.is_on_cooldown(105) is False
    assert rm.cooldown_remaining_seconds() == 0.0


def test_is_on_cooldown_expired_mono_timer_no_op():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 10}})
    rm._cooldown_until_mono = 0.01
    rm.last_result_tick = 0
    assert rm.is_on_cooldown(0) is False
    assert rm._cooldown_until_mono == 0.01


def test_risk_manager_missing_coverage():
    rm = RiskManager({"params": {"stake_min": 1.0}})
    rm.reset_session(500.0)
    assert rm.initial_bankroll == 500.0
    assert rm.total_session_profit == 0.0

    rm.kelly_config = {"fraction": 0.5}
    rm.risk_params = {"payout_estimate": 1.0}
    assert rm.stake_block_reason(100.0, "SYM", conviction=0.9) is None

    rm.begin_cluster(3)
    assert rm.expected_cluster_settlements == 3
    assert rm.cluster_results == {}

    rm.active_contract_ids = [100]
    rm.register_result(10.0, 999, "SYM")
    assert rm.total_session_profit == 0.0
