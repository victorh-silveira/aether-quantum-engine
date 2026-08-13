from unittest.mock import MagicMock

import pytest

from src.domain.risk.consensus_stake_penalty import max_safe_stake_cap
from src.domain.risk.risk_manager import RiskManager
from src.domain.risk.risk_stake_calc import _apply_target_proximity_to_kelly, calculate_stake_for_manager


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


def test_apply_target_proximity_to_kelly_skips_when_stop_win_disabled(kelly_config):
    rm = _mock_rm(kelly_config)
    assert _apply_target_proximity_to_kelly(rm, 25.0, apply_stop_win=False) == pytest.approx(25.0)


def test_calculate_stake_uses_persisted_session_target(kelly_config):
    rm = RiskManager({**kelly_config, "params": {**kelly_config["params"], "compounding_enabled": True}})
    rm.initial_bankroll = 10000.0
    rm.daily_stop_win_target = 88.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=True,
        apply_stop_win=True,
        kwargs={"dl_metrics": {"execute": True}},
    )
    assert stake >= 0.0


def test_calculate_stake_silent_skips_dlambert_log(kelly_config):
    rm = _mock_rm(
        kelly_config,
        pending_loss={"R_10": 100.0},
        consecutive_losses_linear=1,
        dlambert_unit=25.0,
        _recovery_allowed=MagicMock(return_value=True),
    )
    calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=True,
        apply_stop_win=True,
        kwargs={"cycle_id": 3, "dl_metrics": {"val_brier": 0.1, "execute": True}, "order_direction": "PUT"},
    )
    logged = " ".join(str(c) for c in rm.logger.info.call_args_list)
    assert "D'ALEMBERT" not in logged


def test_calculate_stake_for_manager_dlambert_logs(kelly_config):
    rm = _mock_rm(
        kelly_config,
        pending_loss={"R_10": 100.0},
        dlambert_unit=30.0,
        consecutive_losses_linear=1,
        _recovery_allowed=MagicMock(return_value=True),
    )
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=False,
        apply_stop_win=True,
        kwargs={"cycle_id": 3, "dl_metrics": {"val_brier": 0.1, "execute": True}, "order_direction": "PUT"},
    )
    assert stake > 0.0
    audit = getattr(rm, "_last_stake_audit", None)
    assert isinstance(audit, dict)
    assert "DAL_L" in str(audit.get("mode_tag", ""))


def test_calculate_stake_mandatory_weak_entry_uses_full_kelly(kelly_config):
    rm = _mock_rm(kelly_config)
    rm.kelly_config = {
        **kelly_config["kelly"],
        "mandatory_weak_conviction_cap": 0.55,
        "fraction": 0.25,
    }
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "R_10",
        0.70,
        silent=True,
        apply_stop_win=True,
        kwargs={
            "dl_metrics": {"execute": False, "val_brier": 0.1},
            "mandatory_weak_cap": True,
        },
    )
    assert stake >= 10000.0 * 0.0025


def test_calculate_stake_stop_win_kelly_skips_boost_when_conviction_low(kelly_config):
    rm = _mock_rm(
        {
            **kelly_config,
            "large_account_stop_win_pct": 4.0,
            "small_account_threshold": 50.0,
        },
        initial_bankroll=1168.0,
    )
    rm.kelly_config = {
        **kelly_config["kelly"],
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_kelly_min_fraction": 0.42,
        "stop_win_kelly_max_fraction": 1.0,
        "stop_win_kelly_conviction_strong": 0.72,
        "mandatory_weak_conviction_cap": 0.55,
        "fraction": 0.001,
        "cycle_stake_scale_enabled": False,
    }
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
    assert stake >= 1.0


def test_calculate_stake_mandatory_trade_each_cycle(kelly_config):
    rm = _mock_rm(kelly_config)
    rm.kelly_config = {**kelly_config["kelly"], "fraction": 0.05, "kelly_p_floor": 0.55}
    rm.risk_params = {**kelly_config["params"], "stake_min": 1.5}
    rm.effective_win_rate = MagicMock(side_effect=lambda _s, conviction, metrics=None: float(conviction))
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "R_10",
        0.30,
        silent=True,
        apply_stop_win=True,
        kwargs={
            "dl_metrics": {"execute": False, "raw_prob": 0.30},
            "mandatory_trade_each_cycle": True,
        },
    )
    assert stake > 0.0
    assert stake != pytest.approx(50.0)


def test_calculate_stake_dlambert_progresses_without_hard_cap(kelly_config):
    rm = _mock_rm(
        kelly_config,
        pending_loss={"R_10": 400.0},
        consecutive_losses_linear=3,
        dlambert_unit=200.0,
        _recovery_allowed=MagicMock(return_value=True),
    )
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "R_10",
        0.65,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": {"execute": True, "order_direction": "CALL"}},
    )
    assert stake <= 10000.0 * 0.05 + 1e-9


def test_calculate_stake_c0017_bypasses_consensus_and_uses_soft_recovery(kelly_config):
    unit_u = 10.0
    pending = 93.19
    rm = _mock_rm(
        kelly_config,
        pending_loss={"R_10": pending},
        consecutive_losses_linear=3,
        dlambert_unit=unit_u,
        _recovery_allowed=MagicMock(return_value=False),
    )
    rm.kelly_config = {
        **kelly_config["kelly"],
        "consensus_penalty_enabled": True,
        "consensus_max_cut": 0.50,
        "consensus_di_weight": 0.35,
        "consensus_cmo_weight": 0.40,
        "consensus_rsi_weight": 0.25,
        "consensus_entropy_exponent": 2.0,
        "fraction": 0.001,
        "max_stake_pct": 1.0,
    }
    rm.risk_params = {**kelly_config["params"], "stake_min": 1.0}
    rm.effective_win_rate = MagicMock(return_value=0.55)
    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "R_10",
        0.55,
        silent=False,
        apply_stop_win=False,
        kwargs={
            "cycle_id": 17,
            "dl_metrics": {
                "execute": True,
                "call_votes": 1,
                "put_votes": 5,
                "trade_score": 0.55,
                "val_accuracy": 0.40,
                "indicators": {"di_diff": 0.01, "cmo": -0.18, "rsi": 0.62},
            },
            "order_direction": "CALL",
        },
    )
    assert stake > 0.0
    assert stake <= max_safe_stake_cap(10000.0, consecutive_losses_linear=3) + 1e-6
    cover = pending / 0.72 * float(rm.soft_recovery_config.get("cover_multiple", 1.5))
    assert stake <= cover + 1e-2
    audit = getattr(rm, "_last_stake_audit", None)
    assert isinstance(audit, dict)
    assert "DAL_L" in str(audit.get("mode_tag", ""))


def test_calculate_stake_returns_zero_on_soft_veto(kelly_config):
    rm = _mock_rm(kelly_config)
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=True,
        apply_stop_win=True,
        kwargs={"dl_metrics": {"execute": True, "meta_veto_mode": "soft"}},
    )
    assert stake == 0.0
