from unittest.mock import MagicMock

import pytest

from src.domain.risk.dlambert_sizing import dlambert_circuit_breaker, resolve_dlambert_stake
from src.domain.risk.risk_manager import RiskManager
from src.domain.risk.risk_recovery_state import apply_cluster_profit_to_recovery_state
from src.domain.risk.risk_stake_calc import calculate_stake_for_manager


def _base_config(kelly_config):
    return {
        **kelly_config,
        "dlambert": {
            "dlambert_enabled": True,
            "recovery_sizing_conviction": 0.58,
            "recovery_min_conviction": 0.58,
            "recovery_min_val_accuracy": 0.50,
        },
    }


def _risk_manager(kelly_config, *, bankroll=10000.0):
    cfg = _base_config(kelly_config)
    rm = MagicMock()
    rm.config = cfg
    rm.kelly_config = cfg["kelly"]
    rm.dlambert_config = cfg["dlambert"]
    rm.risk_params = {**cfg["params"], "stake_min": 1.0, "payout_estimate": 0.95}
    rm.initial_bankroll = bankroll
    rm.total_session_profit = 0.0
    rm.pending_loss = {}
    rm.active_contract_ids = []
    rm.consecutive_losses_linear = 0
    rm.dlambert_unit = 0.0
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(side_effect=lambda _sym, conv: float(conv))
    rm._recovery_allowed = MagicMock(return_value=True)
    return rm


def test_dlambert_ladder_kelly_loss_loss_win_partial_win_total(kelly_config):
    rm = RiskManager(_base_config(kelly_config))
    rm.initial_bankroll = 10000.0
    rm.kelly_config = {**kelly_config["kelly"], "fraction": 0.10, "dynamic_win_rate": False}
    rm.effective_win_rate = rm.effective_win_rate.__get__(rm, RiskManager)
    rm._recovery_allowed = lambda *_a, **_k: True

    kelly_stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "RDBEAR",
        0.62,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": {"execute": True, "trade_score": 0.62}},
    )
    unit = rm.dlambert_unit
    assert unit > 0.0
    assert kelly_stake == pytest.approx(unit, rel=0.05)

    apply_cluster_profit_to_recovery_state(rm, -float(unit))
    rm.pending_loss["RDBEAR"] = float(unit)
    rm.consecutive_losses_linear = 1

    stake_l1 = calculate_stake_for_manager(
        rm,
        10000.0,
        "RDBEAR",
        0.62,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": {"execute": True, "trade_score": 0.62}},
    )
    assert stake_l1 >= kelly_stake + unit - 1.0

    apply_cluster_profit_to_recovery_state(rm, -float(unit))
    rm.pending_loss["RDBEAR"] += float(unit)
    assert rm.consecutive_losses_linear == 2

    stake_l2 = calculate_stake_for_manager(
        rm,
        10000.0,
        "RDBEAR",
        0.62,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": {"execute": True, "trade_score": 0.62}},
    )
    assert stake_l2 >= kelly_stake + 2 * unit - 1.0

    apply_cluster_profit_to_recovery_state(rm, float(unit) * 0.5)
    assert rm.consecutive_losses_linear == 1

    stake_after_partial = calculate_stake_for_manager(
        rm,
        10000.0,
        "RDBEAR",
        0.62,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": {"execute": True, "trade_score": 0.62}},
    )
    assert stake_after_partial <= stake_l2

    rm.pending_loss = {}
    apply_cluster_profit_to_recovery_state(rm, float(unit))
    assert rm.consecutive_losses_linear == 0

    stake_pure = calculate_stake_for_manager(
        rm,
        10000.0,
        "RDBEAR",
        0.62,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": {"execute": True, "trade_score": 0.62}},
    )
    assert stake_pure == pytest.approx(kelly_stake, rel=0.08)


def test_dlambert_circuit_breaker_halts_stake_on_linear_overflow(kelly_config):
    rm = RiskManager(kelly_config)
    rm.initial_bankroll = 10000.0
    rm.pending_loss["RDBEAR"] = 50.0
    rm.consecutive_losses_linear = 9
    rm.dlambert_unit = 10.0
    rm._recovery_allowed = lambda *_a, **_k: True

    stake = calculate_stake_for_manager(
        rm,
        10000.0,
        "RDBEAR",
        0.61,
        silent=True,
        apply_stop_win=False,
        kwargs={"dl_metrics": {"execute": True, "trade_score": 0.65, "val_accuracy": 0.55}},
    )
    assert stake == 0.0


def test_circuit_breaker_respects_custom_session_drawdown():
    stake, tripped = dlambert_circuit_breaker(
        40.0,
        consecutive_losses_linear=3,
        dlambert_unit=10.0,
        pending_total=300.0,
        max_session_drawdown_u=250.0,
    )
    assert tripped is False
    assert stake == pytest.approx(40.0)


def test_resolve_dlambert_stake_default_drawdown_ceiling_trips():
    class RM:
        dlambert_unit = 9.44
        dlambert_config = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=9.44,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=2,
        pending_total=2078.0,
    )
    assert tag == "D'ALEMBERT_CB"
    assert stake == pytest.approx(0.0)


def test_resolve_dlambert_stake_config_raises_drawdown_ceiling():
    dlambert_config = {
        "dlambert_enabled": True,
        "circuit_breaker_max_session_drawdown_u": 250.0,
    }

    class RM:
        dlambert_unit = 9.44
        dlambert_config = {}

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=9.44,
        dlambert_config=dlambert_config,
        rm=RM(),
        consecutive_losses_linear=2,
        pending_total=2078.0,
    )
    assert tag == "D'ALEMBERT"
    assert stake > 9.44
