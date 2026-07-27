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


def test_calculate_stake_soft_veto_with_senior_trader_conviction(kelly_config):
    rm = _mock_rm(kelly_config)
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=True,
        apply_stop_win=True,
        kwargs={"dl_metrics": {"execute": True, "meta_veto_mode": "soft", "senior_trader_conviction": 0.58}},
    )
    assert stake > 0.0


def test_kelly_no_edge_override_with_senior_trader_conviction(kelly_config):
    """Garante que senior_trader_conviction >= 0.56 isenta gate kelly_no_edge."""
    rm = _mock_rm(kelly_config)
    rm.effective_win_rate = MagicMock(return_value=0.48)
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.52,
        silent=True,
        apply_stop_win=True,
        kwargs={"dl_metrics": {"execute": True, "senior_trader_conviction": 0.58}},
    )
    assert stake > 0.0
