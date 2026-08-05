from unittest.mock import MagicMock

from src.domain.risk.risk_stake_calc import calculate_stake_for_manager


def _mock_rm(kelly_config, **overrides):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = kelly_config["kelly"]
    rm.soft_recovery_config = kelly_config.get("soft_recovery", {})
    rm.dlambert_config = kelly_config.get("dlambert", {})
    rm.risk_params = kelly_config["params"]
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.consecutive_losses_linear = 0
    rm.dlambert_unit = 0.0
    rm.last_loss_stake = 0.0
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._recovery_allowed = MagicMock(return_value=False)
    for key, value in overrides.items():
        setattr(rm, key, value)
    return rm


def test_calculate_stake_ignores_side_eq_blocked_flag(kelly_config):
    rm = _mock_rm(kelly_config)
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=True,
        apply_stop_win=True,
        kwargs={"dl_metrics": {"execute": True, "side_eq_blocked": True}},
    )
    assert stake > 0.0


def test_calculate_stake_returns_zero_on_signal_status_skip(kelly_config):
    rm = _mock_rm(kelly_config)
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=True,
        apply_stop_win=True,
        kwargs={"dl_metrics": {"execute": True, "signal_status": "SKIP"}},
    )
    assert stake == 0.0


def test_calculate_stake_scale_force_explore_sets_regime(kelly_config):
    rm = _mock_rm(kelly_config, pending_loss={}, consecutive_losses_linear=0)
    metrics = {"execute": True, "scale_force_explore": True, "calibrated_prob": 0.60}
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": metrics, "order_direction": "CALL"},
    )
    assert stake > 0.0
    assert metrics["stake_regime"] == "EXPLORE"


def test_calculate_stake_pending_waives_scale_force_explore(kelly_config):
    rm = _mock_rm(kelly_config, pending_loss={"R_10": 20.0}, consecutive_losses_linear=2)
    metrics = {"execute": True, "scale_force_explore": True, "calibrated_prob": 0.60}
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": metrics, "order_direction": "CALL"},
    )
    assert stake > 0.0
    assert metrics["stake_regime"] == "RECOVER"


def test_dlambert_stake_min_zero_f_star_aborts(kelly_config):
    rm = _mock_rm(kelly_config)
    rm._recovery_allowed = MagicMock(return_value=True)
    rm.pending_loss = {"R_10": 0.50}
    rm.consecutive_losses_linear = 1
    rm.dlambert_unit = 0.50
    stake = calculate_stake_for_manager(
        rm,
        10.0,
        "R_10",
        0.50,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": {"execute": True, "f_star": 0.0}},
    )
    assert stake == 0.0
