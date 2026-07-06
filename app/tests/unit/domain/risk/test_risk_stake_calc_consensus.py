from unittest.mock import MagicMock

import pytest

from src.domain.risk.risk_stake_calc import calculate_stake_for_manager


def _attach_dlambert(rm, kelly_config):
    rm.dlambert_config = kelly_config.get("dlambert", {})
    rm.consecutive_losses_linear = 0
    rm.dlambert_unit = 0.0


def test_calculate_stake_consensus_penalty_reduces_stake(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = {
        **kelly_config["kelly"],
        "consensus_penalty_enabled": True,
        "consensus_max_cut": 0.50,
        "consensus_di_weight": 0.35,
        "consensus_cmo_weight": 0.40,
        "fraction": 0.08,
        "max_stake_pct": 1.0,
    }
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 12000.0
    rm.initial_bankroll = 11800.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.80)
    rm._recovery_allowed = MagicMock(return_value=False)
    _attach_dlambert(rm, kelly_config)
    aligned = {
        "execute": True,
        "trade_score": 0.80,
        "raw_prob": 0.78,
        "val_accuracy": 0.70,
        "call_votes": 0,
        "put_votes": 6,
        "indicators": {"di_diff": -0.06, "cmo": -0.71},
    }
    diverged = {**aligned, "call_votes": 1, "put_votes": 5, "indicators": {"di_diff": 0.01, "cmo": -0.18}}
    stake_aligned = calculate_stake_for_manager(
        rm,
        11800.0,
        "RDBULL",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": aligned, "order_direction": "PUT"},
    )
    stake_diverged = calculate_stake_for_manager(
        rm,
        11800.0,
        "RDBEAR",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": diverged, "order_direction": "CALL"},
    )
    assert stake_diverged < stake_aligned


def test_calculate_stake_consensus_floor_uses_session_base_unit(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = {
        **kelly_config["kelly"],
        "consensus_penalty_enabled": True,
        "consensus_max_cut": 0.50,
        "consensus_di_weight": 0.30,
        "consensus_cmo_weight": 0.30,
        "consensus_rsi_weight": 0.25,
        "consensus_entropy_exponent": 2.0,
        "fraction": 0.50,
        "max_stake_pct": 1.0,
    }
    rm.risk_params = {**kelly_config["params"], "stake_min": 1.0}
    rm.stake_max = 12000.0
    rm.initial_bankroll = 11800.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.80)
    rm._recovery_allowed = MagicMock(return_value=False)
    _attach_dlambert(rm, kelly_config)
    metrics = {
        "execute": True,
        "call_votes": 0,
        "put_votes": 6,
        "indicators": {"di_diff": -0.90, "cmo": -0.90, "rsi": 0.15},
    }
    stake = calculate_stake_for_manager(
        rm,
        11800.0,
        "RDBEAR",
        0.80,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": metrics, "order_direction": "CALL"},
    )
    assert stake == pytest.approx(11800.0 * 0.0015)


def test_calculate_stake_consensus_penalty_logs_retention(kelly_config):
    rm = MagicMock()
    rm.config = kelly_config
    rm.kelly_config = {
        **kelly_config["kelly"],
        "consensus_penalty_enabled": True,
        "consensus_max_cut": 0.50,
        "consensus_di_weight": 0.35,
        "consensus_cmo_weight": 0.40,
        "fraction": 0.005,
        "max_stake_pct": 1.0,
    }
    rm.risk_params = kelly_config["params"]
    rm.stake_max = 12000.0
    rm.initial_bankroll = 11800.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.80)
    rm._recovery_allowed = MagicMock(return_value=False)
    _attach_dlambert(rm, kelly_config)
    metrics = {
        "execute": True,
        "call_votes": 1,
        "put_votes": 5,
        "indicators": {"di_diff": 0.01, "cmo": -0.18},
    }
    calculate_stake_for_manager(
        rm,
        11800.0,
        "RDBEAR",
        0.80,
        silent=False,
        apply_stop_win=False,
        kwargs={"dl_metrics": metrics, "order_direction": "CALL", "cycle_id": 0},
    )
    rm.logger.debug.assert_called()
