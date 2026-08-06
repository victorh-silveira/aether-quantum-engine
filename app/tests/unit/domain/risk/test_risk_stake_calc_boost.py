from unittest.mock import MagicMock

from src.domain.risk.risk_stake_calc import calculate_stake_for_manager


def _attach_dlambert(rm, kelly_config):
    rm.dlambert_config = kelly_config.get("dlambert", {})
    rm.soft_recovery_config = kelly_config.get("soft_recovery", {})
    rm.consecutive_losses_linear = 0
    rm.dlambert_unit = 0.0
    return rm


def test_calculate_stake_stop_win_kelly_boosts_from_raw_when_score_zero(kelly_config):
    rm = MagicMock()
    rm.config = {
        **kelly_config,
        "large_account_stop_win_pct": 4.0,
        "small_account_threshold": 50.0,
        "orchestrator": {"cycle_interval_seconds": 300},
    }
    rm.kelly_config = {
        **kelly_config["kelly"],
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_kelly_min_fraction": 0.42,
        "stop_win_kelly_max_fraction": 1.0,
        "stop_win_kelly_conviction_strong": 0.72,
        "stop_win_kelly_cycles_target": 1.0,
        "stake_conviction_min_raw": 0.51,
        "fraction": 0.001,
        "max_stake_pct": 0.01,
        "mandatory_weak_max_stake_pct": 0.006,
        "cycle_stake_scale_enabled": False,
    }
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 12000.0
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    _attach_dlambert(rm, kelly_config)
    rm._recovery_allowed = MagicMock(return_value=False)
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "OTC_SPC",
        0.0,
        silent=True,
        apply_stop_win=True,
        kwargs={
            "dl_metrics": {
                "execute": False,
                "trade_score": 0.0,
                "raw_prob": 0.51,
                "val_accuracy": 0.59,
                "live_n": 40,
                "live_wr": 0.55,
            },
            "mandatory_weak_cap": True,
        },
    )
    assert stake > 20.0


def test_calculate_stake_dlambert_recovery_adds_linear_unit(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = kelly_config["kelly"]
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 12000.0
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {"OTC_SPC": 200.0}
    rm.active_contract_ids = []
    rm.last_loss_stake = 50.0
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    _attach_dlambert(rm, kelly_config)
    rm.dlambert_unit = 30.0
    rm.consecutive_losses_linear = 2
    rm._recovery_allowed = MagicMock(return_value=True)
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "OTC_SPC",
        0.0,
        silent=True,
        apply_stop_win=True,
        kwargs={
            "dl_metrics": {
                "execute": False,
                "trade_score": 0.0,
                "raw_prob": 0.51,
                "val_accuracy": 0.55,
                "consensus_strength": 0.2,
            },
        },
    )
    assert stake > 40.0


def test_calculate_stake_mandatory_weak_boost_unlimited(kelly_config):
    rm = MagicMock()
    rm.config = {
        **kelly_config,
        "large_account_stop_win_pct": 4.0,
        "small_account_threshold": 50.0,
        "orchestrator": {"cycle_interval_seconds": 300},
    }
    rm.kelly_config = {
        **kelly_config["kelly"],
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_kelly_min_fraction": 0.42,
        "stop_win_kelly_max_fraction": 1.0,
        "stop_win_kelly_conviction_strong": 0.72,
        "stop_win_kelly_cycles_target": 1.0,
        "mandatory_weak_max_stake_pct": 0.006,
        "mandatory_weak_conviction_cap": 0.55,
        "fraction": 0.001,
        "max_stake_pct": 0.01,
        "cycle_stake_scale_enabled": True,
        "cycle_stake_baseline_seconds": 60,
        "cycle_stake_exponent": 0.55,
    }
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 12000.0
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    _attach_dlambert(rm, kelly_config)
    rm._recovery_allowed = MagicMock(return_value=False)
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "OTC_SPC",
        0.50,
        silent=True,
        apply_stop_win=True,
        kwargs={
            "dl_metrics": {"execute": False, "val_brier": 0.1, "live_n": 40, "live_wr": 0.55},
            "mandatory_weak_cap": True,
        },
    )
    assert stake > 20.0


def test_calculate_stake_stop_win_kelly_boosts_when_dl_approved(kelly_config):
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
        "fraction": 0.05,
        "max_stake_pct": 0.02,
    }
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 12000.0
    rm.initial_bankroll = 1168.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    _attach_dlambert(rm, kelly_config)
    rm._recovery_allowed = MagicMock(return_value=False)
    stake = calculate_stake_for_manager(
        rm,
        1168.0,
        "OTC_SPC",
        0.55,
        silent=True,
        apply_stop_win=True,
        kwargs={"dl_metrics": {"execute": True, "val_brier": 0.1, "live_n": 40, "live_wr": 0.55}},
    )
    assert stake > 1.5


def test_calculate_stake_stop_win_kelly_silent_skips_boost_log(kelly_config):
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
        "fraction": 0.001,
        "max_stake_pct": 0.004,
    }
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 12000.0
    rm.initial_bankroll = 1168.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    _attach_dlambert(rm, kelly_config)
    rm._recovery_allowed = MagicMock(return_value=False)
    calculate_stake_for_manager(
        rm,
        1168.0,
        "OTC_SPC",
        0.50,
        silent=True,
        apply_stop_win=True,
        kwargs={"dl_metrics": {"execute": False, "val_brier": 0.1}},
    )
    logged = " ".join(str(c) for c in rm.logger.info.call_args_list)
    assert "STOP WIN KELLY" not in logged
