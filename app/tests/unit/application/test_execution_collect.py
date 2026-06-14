from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.execution_symbols import (
    has_recovery_hedge_candidate,
    inject_recovery_hedge_candidates,
)
from src.application.services.orchestrator.execution_collect import (
    _cluster_entry_eligible,
    _gather_cluster_candidates,
    apply_recovery_hedge_to_candidates,
    collect_cluster_orders,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, HEDGE_PEER_SYMBOL, PAIR


def test_inject_recovery_hedge_early_returns():
    base = [(PAIR, TradeDirection.CALL, {"execute": True})]
    assert (
        inject_recovery_hedge_candidates(
            base,
            {},
            last_loss_symbol=None,
            last_loss_direction=None,
        )
        == base
    )
    present = [(HEDGE_PEER_SYMBOL, TradeDirection.PUT, {"execute": True})]
    assert (
        inject_recovery_hedge_candidates(
            present,
            {HEDGE_PEER_SYMBOL: {"direction": TradeDirection.CALL, "metrics": {}}},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        )
        == present
    )
    assert (
        inject_recovery_hedge_candidates(
            base,
            {HEDGE_PEER_SYMBOL: {"direction": None, "metrics": {}}},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        )
        == base
    )
    assert has_recovery_hedge_candidate(base, last_loss_symbol=None, last_loss_direction=None)


def test_apply_recovery_hedge_keeps_same_direction_candidates():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        ),
        _active_cycle_id=2,
    )
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [(PAIR, TradeDirection.CALL, {"execute": True})]
    result = apply_recovery_hedge_to_candidates(
        exec_mgr,
        candidates,
        {},
        cid="C0002",
    )
    assert result == candidates


def test_apply_recovery_hedge_keeps_candidates_for_market_ranking():
    orch = SimpleNamespace(
        config={
            "risk_management": {"kelly": {}},
            "orchestrator": {"execution": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 5.0},
            last_loss_symbol=PAIR,
            last_loss_direction="PUT",
        ),
        _active_cycle_id=3,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [ANCHOR, PAIR],
    )
    candidates = [(ANCHOR, TradeDirection.CALL, {"execute": True})]
    result = apply_recovery_hedge_to_candidates(
        exec_mgr,
        candidates,
        {},
        cid="C0003",
        mandatory=True,
    )
    assert result == candidates


def test_apply_recovery_hedge_passthrough_without_pending():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        risk_manager=SimpleNamespace(pending_loss={}, last_loss_symbol=None, last_loss_direction=None),
        _active_cycle_id=1,
    )
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [(ANCHOR, TradeDirection.CALL, {})]
    assert apply_recovery_hedge_to_candidates(exec_mgr, candidates, {}, cid="C0001") == candidates


def test_collect_cluster_orders_recovery_executes_best_available_signal():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True}},
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="PUT",
            consecutive_losses=0,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
        ),
        _active_cycle_id=10,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [ANCHOR],
    )
    decisions = {
        ANCHOR: {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": True,
                "deploy_ok": True,
                "val_accuracy": 0.55,
                "conviction": 0.60,
                "raw_prob": 0.58,
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == ANCHOR
    assert orders[0][1] == TradeDirection.CALL


def test_collect_cluster_orders_empty_after_recovery_skip():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        ),
        _active_cycle_id=9,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: False,
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {
        PAIR: {"direction": TradeDirection.CALL, "metrics": {"raw_prob": 0.4, "execute": True}},
    }
    assert collect_cluster_orders(exec_mgr, decisions) == []


def test_apply_recovery_hedge_keeps_pool_when_direction_differs_from_loss():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="PUT",
        ),
        _active_cycle_id=7,
    )
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [(ANCHOR, TradeDirection.CALL, {"execute": True})]
    result = apply_recovery_hedge_to_candidates(
        exec_mgr,
        candidates,
        {},
        cid="C0007",
    )
    assert result == candidates


def test_collect_cluster_orders_skips_execute_false_in_recovery():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        ),
        _active_cycle_id=9,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: False,
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {
        PAIR: {"direction": TradeDirection.CALL, "metrics": {"raw_prob": 0.4, "execute": False}},
    }
    assert collect_cluster_orders(exec_mgr, decisions) == []


def test_collect_cluster_orders_mandatory_does_not_skip_recovery_without_hedge():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        ),
        _active_cycle_id=9,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {
        PAIR: {"direction": TradeDirection.CALL, "metrics": {"raw_prob": 0.4, "execute": True}},
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == PAIR


def test_cluster_entry_recovery_rejects_mandatory_weak_bypass():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "execute": False,
            "trade_score": 0.51,
            "val_accuracy": 0.59,
            "raw_prob": 0.49,
            "deploy_ok": True,
        },
    }
    assert not _cluster_entry_eligible(
        entry,
        mandatory=True,
        recovery_active=True,
        recovery_cfg={},
        min_signal=0.50,
        min_val=0.50,
    )


def test_gather_cluster_candidates_skips_unbuildable_direction():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[PAIR],
        _active_cycle_id=1,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {PAIR: {"direction": None, "metrics": {"execute": True}}}
    candidates = _gather_cluster_candidates(
        exec_mgr,
        decisions,
        mandatory=False,
        recovery_active=False,
        recovery_cfg={},
        cid="C0001",
        min_signal=0.45,
        min_val=0.0,
    )
    assert candidates == []
