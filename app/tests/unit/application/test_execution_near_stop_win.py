from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.application.services.orchestrator.execution_near_stop_win import (
    best_blocked_signal_strength,
    decisions_all_dl_blocked,
    near_stop_win_mandatory_pause,
    should_pause_weak_mandatory,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


def test_decisions_all_dl_blocked():
    assert decisions_all_dl_blocked({}) is True
    assert decisions_all_dl_blocked({ANCHOR: {"metrics": {"execute": False}}, PAIR: {"metrics": {"execute": False}}})
    assert not decisions_all_dl_blocked({ANCHOR: {"metrics": {"execute": True}}})


def test_should_pause_false_when_recovery_active():
    orch = SimpleNamespace(
        config={"risk_management": {"kelly": {}}},
        risk_manager=SimpleNamespace(pending_loss={ANCHOR: 1.0}),
    )
    exec_mgr = SimpleNamespace(orch=orch)
    assert should_pause_weak_mandatory(exec_mgr, {}, recovery_active=True) is False


def test_should_pause_false_when_dl_has_executable_symbol():
    orch = SimpleNamespace(
        config={"risk_management": {"kelly": {"mandatory_min_trade_score": 0.45}}},
        risk_manager=SimpleNamespace(
            pending_loss={},
            total_session_profit=0.0,
            initial_bankroll=1000.0,
        ),
    )
    exec_mgr = SimpleNamespace(orch=orch)
    decisions = {ANCHOR: {"metrics": {"execute": True}}}
    assert should_pause_weak_mandatory(exec_mgr, decisions, recovery_active=False) is False


def test_near_stop_win_mandatory_pause_ignores_recovery():
    rm = SimpleNamespace(pending_loss={ANCHOR: 4.0}, total_session_profit=50.0, initial_bankroll=1000.0)
    risk_cfg = {"small_account_threshold": 50.0, "large_account_stop_win_pct": 4.0}
    kelly_cfg = {"near_stop_win_mandatory_pause_fraction": 0.9}
    assert near_stop_win_mandatory_pause(rm, risk_cfg, kelly_cfg) is False


def test_best_blocked_signal_strength_ignores_hard_blocked():
    strength = best_blocked_signal_strength(
        {
            ANCHOR: {"metrics": {"execute": False, "gate_reason": "data", "trade_score": 0.9}},
            PAIR: {"metrics": {"execute": False, "trade_score": 0.52, "raw_prob": 0.51}},
        }
    )
    assert strength == pytest.approx(0.52, abs=0.01)


def test_collect_cluster_orders_pauses_on_weak_blocked_signal():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True}},
            "risk_management": {
                "small_account_threshold": 50.0,
                "large_account_stop_win_pct": 4.0,
                "kelly": {"mandatory_min_trade_score": 0.45, "near_stop_win_mandatory_pause_fraction": 0.5},
            },
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
            total_session_profit=0.0,
            initial_bankroll=1000.0,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
        ),
        _active_cycle_id=9,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [ANCHOR, PAIR],
    )
    decisions = {
        ANCHOR: {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.30, "val_accuracy": 0.38},
        },
        PAIR: {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": False, "trade_score": 0.20, "val_accuracy": 0.38},
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert orders == []
    assert should_pause_weak_mandatory(exec_mgr, decisions, recovery_active=False) is True


def test_near_stop_win_mandatory_pause_disabled_when_fraction_zero():
    rm = SimpleNamespace(pending_loss={}, total_session_profit=50.0, initial_bankroll=1000.0)
    risk_cfg = {"small_account_threshold": 50.0, "large_account_stop_win_pct": 4.0}
    kelly_cfg = {"near_stop_win_mandatory_pause_fraction": 0.0}
    assert near_stop_win_mandatory_pause(rm, risk_cfg, kelly_cfg) is False


def test_near_stop_win_mandatory_pause_false_when_target_zero():
    rm = SimpleNamespace(pending_loss={}, total_session_profit=50.0, initial_bankroll=1000.0)
    risk_cfg = {"small_account_threshold": 50.0, "large_account_stop_win_pct": 0.0}
    kelly_cfg = {"near_stop_win_mandatory_pause_fraction": 0.9}
    assert near_stop_win_mandatory_pause(rm, risk_cfg, kelly_cfg) is False


def test_near_stop_win_mandatory_pause_when_near_target():
    rm = SimpleNamespace(pending_loss={}, total_session_profit=38.0, initial_bankroll=1000.0)
    risk_cfg = {"small_account_threshold": 50.0, "large_account_stop_win_pct": 4.0}
    kelly_cfg = {"near_stop_win_mandatory_pause_fraction": 0.9}
    assert near_stop_win_mandatory_pause(rm, risk_cfg, kelly_cfg) is True


def test_collect_cluster_orders_pauses_weak_mandatory_near_stop_win():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True}},
            "risk_management": {
                "small_account_threshold": 50.0,
                "large_account_stop_win_pct": 4.0,
                "kelly": {"near_stop_win_mandatory_pause_fraction": 0.9},
            },
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
            total_session_profit=42.0,
            initial_bankroll=1126.82,
        ),
        _active_cycle_id=40,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [ANCHOR, PAIR],
    )
    decisions = {
        ANCHOR: {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": False, "trade_score": 0.0, "val_accuracy": 0.56, "raw_prob": 0.41},
        },
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.52, "val_accuracy": 0.47, "raw_prob": 0.51},
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert should_pause_weak_mandatory(exec_mgr, decisions, recovery_active=False) is True
    assert not any("near_stop_win_weak_mandatory" in str(c) for c in exec_mgr.logger.info.call_args_list)
