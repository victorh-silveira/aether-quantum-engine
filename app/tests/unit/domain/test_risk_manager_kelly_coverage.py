"""Testes de cobertura para RiskManager Kelly."""

import time
from unittest.mock import patch

from src.domain.risk.risk_manager import RiskManager


def test_risk_manager_rolling_wins_cap():
    """Cobre o limite de amostras no histórico de vitórias."""
    rm = RiskManager({})
    for _ in range(110):
        rm.record_trade_outcome("SYM", won=True)
    wr, n = rm.get_wr_rolling_stats("SYM")
    assert n == 100


def test_risk_manager_cooldown_active():
    """Cobre o estado de cooldown ativo."""
    rm = RiskManager({"params": {"entry_cooldown_ticks": 10}})
    rm.last_result_tick = 100
    assert rm.is_on_cooldown(105) is True
    assert rm.is_on_cooldown(115) is False


def test_risk_manager_high_conviction_shorter_cooldown():
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
    assert rm.effective_cooldown_ticks() == 6
    assert rm.is_on_cooldown(105) is True
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
    assert rm.effective_cooldown_ticks() == 8


def test_risk_manager_seconds_cooldown_active():
    rm = RiskManager({"params": {"entry_cooldown_seconds": 30}})
    rm.register_entry_conviction(0.7)
    rm._arm_cooldown_timer()
    assert rm._uses_seconds_cooldown() is True
    assert rm.is_on_cooldown(1) is True


def test_cooldown_span_uses_candle_interval_without_tick_seconds_override():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 3}})
    rm.set_candle_interval_seconds(120)
    rm.current_cooldown_ticks = 3
    assert rm._cooldown_span_seconds() == 360.0


def test_is_on_cooldown_true_while_mono_timer_active():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 5}})
    rm._cooldown_until_mono = time.monotonic() + 120.0
    assert rm.is_on_cooldown(0) is True


def test_cooldown_span_uses_tick_seconds_override():
    rm = RiskManager(
        {
            "params": {
                "entry_cooldown_ticks": 4,
                "entry_cooldown_tick_seconds": 15,
            }
        }
    )
    rm.current_cooldown_ticks = 4
    assert rm._cooldown_span_seconds() == 60.0


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
    assert rm.stake_block_reason(50.0, "OTC_FCHI") == "stop_win"


def test_arm_cooldown_zero_ticks_clears_timer():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 0}})
    rm._cooldown_until_mono = 999.0
    rm._arm_cooldown_timer()
    assert rm._cooldown_until_mono == 0.0
    assert rm.is_on_cooldown(99) is False


def test_stop_win_aggressive_stake_no_boost_when_target_hit():
    rm = RiskManager(
        {
            "small_account_stop_win": 10.0,
            "kelly": {"stop_win_aggressive": True},
            "params": {"payout_estimate": 1.0},
        }
    )
    rm.set_initial_bankroll(50.0)
    rm.total_session_profit = 12.0
    assert rm._apply_stop_win_aggressive_stake(48.0, 2.0) == 2.0


def test_stop_win_aggressive_stake_boost():
    rm = RiskManager(
        {
            "small_account_stop_win": 10.0,
            "small_account_threshold": 100.0,
            "kelly": {
                "stop_win_aggressive": True,
                "stop_win_stake_multiplier": 1.5,
                "stop_win_stake_cap_pct": 0.12,
            },
            "params": {"payout_estimate": 1.0, "stake_min": 1.0},
        }
    )
    rm.set_initial_bankroll(50.0)
    rm.total_session_profit = 2.0
    boosted = rm._apply_stop_win_aggressive_stake(48.0, 1.0)
    assert boosted >= 5.0
    assert boosted <= 48.0 * 0.12


def test_stake_block_reason_kelly_no_edge():
    rm = RiskManager(
        {
            "kelly": {"fraction": 0.01},
            "params": {"payout_estimate": 0.5, "stake_min": 1.0},
        }
    )
    rm.set_initial_bankroll(100.0)
    assert rm.stake_block_reason(100.0, "OTC_FCHI", conviction=0.05) == "kelly_no_edge"


def test_cooldown_mono_expiry_falls_back_to_ticks():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 10}})
    rm.set_candle_interval_seconds(60)
    rm._cooldown_until_mono = 0.0
    rm.last_result_tick = 100
    rm.current_cooldown_ticks = 10
    assert rm.is_on_cooldown(105) is True
    assert rm.cooldown_remaining_seconds() == 0.0


def test_is_on_cooldown_clears_expired_mono_timer():
    rm = RiskManager({"params": {"entry_cooldown_ticks": 10}})
    rm._cooldown_until_mono = 0.01
    rm.last_result_tick = 0
    with patch("src.domain.risk.risk_cooldown.time.monotonic", return_value=1.0):
        assert rm.is_on_cooldown(0) is False
    assert rm._cooldown_until_mono == 0.0
