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
    rm.pending_loss = {"R_50": 100.0}
    rm.consecutive_losses = 1
    rm.last_martingale_stake = 0.0
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._apply_stop_win_aggressive_stake = MagicMock(side_effect=lambda _b, raw, **_: raw)
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
    rm._apply_stop_win_aggressive_stake = MagicMock(side_effect=lambda _b, raw, **_: raw)
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


def test_calculate_stake_logs_martingale_block(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = kelly_config["kelly"]
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 10000.0
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {"R_50": 100.0}
    rm.active_contract_ids = []
    rm.martingale_native = True
    rm.martingale_block_repeat_loss = False
    rm.consecutive_losses = 1
    rm.last_martingale_stake = 0.0
    rm.last_loss_symbol = None
    rm.last_loss_direction = None
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._apply_stop_win_aggressive_stake = MagicMock(side_effect=lambda _b, raw, **_: raw)
    rm._martingale_allowed = MagicMock(return_value=False)
    calculate_stake_for_manager(
        rm,
        5000.0,
        "R_50",
        0.3,
        silent=True,
        apply_stop_win=True,
        kwargs={"cycle_id": 2, "dl_metrics": {"gate_reason": "deploy"}},
    )
    logged = " ".join(str(c) for c in rm.logger.info.call_args_list)
    assert "Martingale bloqueado" in logged


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
    rm._apply_stop_win_aggressive_stake = MagicMock(side_effect=lambda _b, raw, **_: raw)
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
