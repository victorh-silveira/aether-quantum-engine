from unittest.mock import MagicMock

from src.domain.risk.risk_stake_calc import calculate_stake_for_manager


def test_calculate_stake_silent_skips_martingale_log(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = kelly_config["kelly"]
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 10000.0
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {"RDBULL": 100.0}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._apply_stop_win_aggressive_stake = MagicMock(side_effect=lambda _b, raw, **_: raw)
    rm._martingale_allowed = MagicMock(return_value=True)
    calculate_stake_for_manager(
        rm,
        5000.0,
        "RDBULL",
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
    rm.pending_loss = {"RDBULL": 100.0}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._apply_stop_win_aggressive_stake = MagicMock(side_effect=lambda _b, raw, **_: raw)
    rm._martingale_allowed = MagicMock(return_value=True)
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "RDBULL",
        0.6,
        silent=False,
        apply_stop_win=True,
        kwargs={"cycle_id": 3, "dl_metrics": {"val_brier": 0.1}, "order_direction": "PUT"},
    )
    assert stake > 100.0
    logged = " ".join(str(c) for c in rm.logger.info.call_args_list)
    assert "MARTINGALE" in logged


def test_calculate_stake_logs_martingale_block(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = kelly_config["kelly"]
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 10000.0
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {"RDBULL": 100.0}
    rm.active_contract_ids = []
    rm.recovery_threshold = 0.72
    rm.recovery_martingale_min_conviction = 0.45
    rm.martingale_force_on_pending_loss = False
    rm.last_loss_symbol = None
    rm.last_loss_direction = None
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._apply_stop_win_aggressive_stake = MagicMock(side_effect=lambda _b, raw, **_: raw)
    rm._martingale_allowed = MagicMock(return_value=False)
    calculate_stake_for_manager(
        rm,
        5000.0,
        "RDBULL",
        0.3,
        silent=True,
        apply_stop_win=True,
        kwargs={"cycle_id": 2, "dl_metrics": {"gate_reason": "deploy"}},
    )
    logged = " ".join(str(c) for c in rm.logger.info.call_args_list)
    assert "Martingale bloqueado" in logged
