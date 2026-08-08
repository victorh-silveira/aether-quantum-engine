from unittest.mock import MagicMock

import pytest

from src.domain.risk.risk_stake_calc import calculate_stake_for_manager


def _attach_dlambert(rm, kelly_config):
    rm.dlambert_config = kelly_config.get("dlambert", {})
    rm.soft_recovery_config = kelly_config.get("soft_recovery", {})
    rm.consecutive_losses_linear = 0
    rm.dlambert_unit = 0.0


def _base_rm(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = {
        **kelly_config["kelly"],
        "consensus_penalty_enabled": True,
        "consensus_max_cut": 0.50,
        "fraction": 0.001,
        "max_stake_pct": 1.0,
        "stop_win_kelly_enabled": False,
    }
    rm.risk_params = {**kelly_config["params"], "stake_min": 1.0}
    rm.stake_max = 12000.0
    rm.initial_bankroll = 11000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.80)
    rm._recovery_allowed = MagicMock(return_value=False)
    _attach_dlambert(rm, kelly_config)
    return rm


def test_calculate_stake_neutral_regime_uses_dynamic_bankroll_base(kelly_config):
    rm = _base_rm(kelly_config)
    rm.effective_win_rate = MagicMock(side_effect=lambda _s, conviction, metrics=None: float(conviction))
    metrics = {
        "execute": True,
        "trade_score": 0.80,
        "raw_prob": 0.78,
        "edge_expectancy": "NO_EDGE_NEUTRAL",
        "edge_zscore": 0.1,
        "call_votes": 0,
        "put_votes": 6,
        "indicators": {"di_diff": -0.06, "cmo": -0.71},
    }
    stake = calculate_stake_for_manager(
        rm,
        11000.0,
        "R_10",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": metrics, "order_direction": "PUT"},
    )
    assert stake == pytest.approx(27.5)
    assert metrics.get("session_base_unit") == pytest.approx(27.5)


def test_calculate_stake_turbo_edge_doubles_final_stake(kelly_config):
    rm = _base_rm(kelly_config)
    base_metrics = {
        "execute": True,
        "trade_score": 0.80,
        "raw_prob": 0.78,
        "edge_expectancy": "WIN_EXPECTED",
        "edge_zscore": 0.9,
        "live_n": 32,
        "live_brier": 0.15,
        "call_votes": 0,
        "put_votes": 6,
        "indicators": {"di_diff": -0.06, "cmo": -0.71},
    }
    turbo_metrics = {**base_metrics, "edge_zscore": 1.8}
    stake_base = calculate_stake_for_manager(
        rm,
        11000.0,
        "R_10",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": dict(base_metrics), "order_direction": "PUT"},
    )
    stake_turbo = calculate_stake_for_manager(
        rm,
        11000.0,
        "R_10",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": turbo_metrics, "order_direction": "PUT"},
    )
    assert stake_base == pytest.approx(27.5)
    assert stake_turbo == pytest.approx(min(stake_base * 2.0, 11000.0 * 0.035))
    assert turbo_metrics.get("consensus_turbo_edge_active") is True


def test_calculate_stake_c0007_turbo_on_clean_recovery_base(kelly_config):
    rm = _base_rm(kelly_config)
    rm.pending_loss = {"R_10": 36.72}
    rm.consecutive_losses_linear = 1
    rm.dlambert_unit = 17.89
    rm.last_loss_stake = 36.72
    rm._recovery_allowed = MagicMock(return_value=True)
    rm.risk_params = {**kelly_config["params"], "stake_min": 1.0, "payout_estimate": 0.95}
    neutral_metrics = {
        "execute": True,
        "trade_score": 0.80,
        "raw_prob": 0.78,
        "edge_expectancy": "NO_EDGE_NEUTRAL",
        "edge_zscore": 0.1,
        "live_n": 32,
        "live_brier": 0.15,
        "call_votes": 0,
        "put_votes": 6,
        "indicators": {"di_diff": -0.06, "cmo": -0.71},
    }
    turbo_metrics = {
        **neutral_metrics,
        "edge_expectancy": "WIN_EXPECTED",
        "edge_zscore": 1.8,
    }
    bankroll = 11926.67
    stake_neutral = calculate_stake_for_manager(
        rm,
        bankroll,
        "R_10",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": dict(neutral_metrics), "order_direction": "PUT"},
    )
    assert stake_neutral == pytest.approx(37.34, rel=1e-2)
    stake_turbo = calculate_stake_for_manager(
        rm,
        bankroll,
        "R_10",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": turbo_metrics, "order_direction": "PUT"},
    )
    soft_pct = float(rm.soft_recovery_config.get("max_safe_stake_pct", 0.05))
    assert stake_turbo == pytest.approx(min(stake_neutral * 2.0, bankroll * soft_pct), rel=1e-2)
    assert turbo_metrics.get("consensus_turbo_edge_active") is True


def test_calculate_stake_d_squeeze_keeps_floor_not_turbo(kelly_config):
    rm = _base_rm(kelly_config)
    metrics = {
        "execute": True,
        "trade_score": 0.52,
        "meta_squeeze_downgrade": True,
        "consensus_stake_floor": True,
        "edge_expectancy": "WIN_EXPECTED",
        "edge_zscore": 2.0,
        "call_votes": 0,
        "put_votes": 6,
        "indicators": {"di_diff": -0.06, "cmo": -0.71},
    }
    stake = calculate_stake_for_manager(
        rm,
        11000.0,
        "R_10",
        0.52,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": metrics, "order_direction": "PUT"},
    )
    assert stake == pytest.approx(1.0)
    assert metrics.get("consensus_turbo_edge_active") is not True
