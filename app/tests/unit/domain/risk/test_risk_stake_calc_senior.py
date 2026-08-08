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


def test_thin_edge_still_sizes_via_kelly_p_floor(kelly_config):
    """Edge fino: p floor garante f*>0 e stake>0 sem inventar 0.5% da banca."""
    rm = _mock_rm(kelly_config)
    rm.kelly_config = {**kelly_config["kelly"], "kelly_p_floor": 0.55, "fraction": 0.3, "max_stake_pct": 0.05}
    rm.effective_win_rate = MagicMock(side_effect=lambda _s, conviction, metrics=None: float(conviction))
    metrics = {"execute": True, "calibrated_prob": 0.51, "trade_score": 0.51}
    bankroll = 10000.0
    stake = calculate_stake_for_manager(
        rm,
        bankroll,
        "R_10",
        0.51,
        silent=True,
        apply_stop_win=True,
        kwargs={"dl_metrics": metrics, "order_direction": "CALL"},
    )
    assert stake > 0.0
    assert stake >= bankroll * 0.0025 - 1e-6
    assert float(metrics.get("kelly_side_p", 0.0)) >= 0.55


def test_soft_veto_still_blocks_without_senior(kelly_config):
    rm = _mock_rm(kelly_config)
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=True,
        apply_stop_win=True,
        kwargs={"dl_metrics": {"execute": True, "meta_veto_mode": "soft", "senior_trader_conviction": 0.40}},
    )
    assert stake == 0.0
