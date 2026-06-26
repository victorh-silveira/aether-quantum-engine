from unittest.mock import MagicMock

from src.domain.risk.risk_stake_calc import (
    _apply_mandatory_weak_cap,
    calculate_stake_for_manager,
)


def test_calculate_stake_silent_skips_martingale_log(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = kelly_config["kelly"]
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 10000.0
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {"R_50": 100.0}
    rm.consecutive_losses = 1
    rm.last_martingale_stake = 0.0
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._martingale_allowed = MagicMock(return_value=True)
    calculate_stake_for_manager(
        rm,
        5000.0,
        "R_50",
        0.6,
        silent=True,
        apply_stop_win=True,
        kwargs={"cycle_id": 3, "dl_metrics": {"val_brier": 0.1}, "order_direction": "PUT"},
    )
    logged = " ".join(str(c) for c in rm.logger.info.call_args_list)
    assert "MARTINGALE" not in logged


def test_calculate_stake_for_manager_martingale_logs(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = kelly_config["kelly"]
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 10000.0
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {"R_50": 100.0}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._martingale_allowed = MagicMock(return_value=True)
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_50",
        0.6,
        silent=False,
        apply_stop_win=True,
        kwargs={"cycle_id": 3, "dl_metrics": {"val_brier": 0.1}, "order_direction": "PUT"},
    )
    assert stake > 100.0
    logged = " ".join(str(c) for c in rm.logger.info.call_args_list)
    assert "MARTINGALE" in logged


def test_calculate_stake_mandatory_weak_cap(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = {
        **kelly_config["kelly"],
        "mandatory_weak_max_stake_pct": 0.004,
        "mandatory_weak_conviction_cap": 0.55,
    }
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 10000.0
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._martingale_allowed = MagicMock(return_value=False)
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "R_50",
        0.70,
        silent=True,
        apply_stop_win=True,
        kwargs={
            "dl_metrics": {"execute": False, "val_brier": 0.1},
            "mandatory_weak_cap": True,
        },
    )
    assert stake <= 10000.0 * 0.004 + 0.01


def test_calculate_stake_stop_win_kelly_skips_boost_when_conviction_low(kelly_config):
    rm = MagicMock()
    rm.config = {
        **kelly_config,
        "large_account_stop_win_pct": 4.0,
        "small_account_threshold": 50.0,
    }
    rm.kelly_config = {
        **kelly_config["kelly"],
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_kelly_min_fraction": 0.42,
        "stop_win_kelly_max_fraction": 1.0,
        "stop_win_kelly_conviction_strong": 0.72,
        "mandatory_weak_max_stake_pct": 0.004,
        "mandatory_weak_conviction_cap": 0.55,
        "fraction": 0.001,
        "max_stake_pct": 0.004,
        "cycle_stake_scale_enabled": False,
    }
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 12000.0
    rm.initial_bankroll = 1168.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._martingale_allowed = MagicMock(return_value=False)
    stake = calculate_stake_for_manager(
        rm,
        1168.0,
        "R_10",
        0.40,
        silent=True,
        apply_stop_win=True,
        kwargs={
            "dl_metrics": {"execute": False, "val_brier": 0.1},
            "mandatory_weak_cap": True,
        },
    )
    assert stake <= 1168.0 * 0.004 + 0.02


def test_calculate_stake_mandatory_trade_each_cycle(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = {
        **kelly_config["kelly"],
        "fraction": 0.05,
    }
    rm.risk_params = {
        **kelly_config["params"],
        "stake_min": 1.5,
    }
    rm.stake_max = 10000.0
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.20)  # Força Kelly < 0 (sem edge)
    rm._martingale_allowed = MagicMock(return_value=False)
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "R_50",
        0.30,  # Baixa convicção
        silent=True,
        apply_stop_win=True,
        kwargs={
            "dl_metrics": {"execute": False, "raw_prob": 0.30},
            "mandatory_trade_each_cycle": True,
        },
    )
    # Mesmo sem edge de Kelly e com baixa convicção, deve alocar stake_min (1.5) por ser execução mandatória
    assert stake == 1.5


def test_apply_mandatory_weak_cap_zero_pct():
    result = _apply_mandatory_weak_cap(
        50.0,
        10000.0,
        {"mandatory_weak_max_stake_pct": 0.0, "max_stake_pct": 0.0},
        {"mandatory_weak_cap": True},
        martingale_active=False,
    )
    assert result == 50.0
