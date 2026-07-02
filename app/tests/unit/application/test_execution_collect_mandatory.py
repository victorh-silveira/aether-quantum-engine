from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.application.services.execution_symbols_recovery import recovery_blocked_symbols
from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.application.services.orchestrator.execution_collect_helpers import resolve_mandatory_ultimate_candidate
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


def test_collect_cluster_orders_mandatory_fallback_after_recovery_filter():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True}},
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={ANCHOR: 4.64},
            last_loss_symbol=ANCHOR,
            last_loss_direction="PUT",
            consecutive_losses=0,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
        ),
        _active_cycle_id=11,
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
            "metrics": {
                "execute": False,
                "trade_score": 0.65,
                "val_accuracy": 0.50,
                "raw_prob": 0.47,
                "deploy_ok": True,
            },
        },
        PAIR: {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": False,
                "trade_score": 0.58,
                "val_accuracy": 0.52,
                "raw_prob": 0.44,
                "deploy_ok": True,
            },
        },
    }
    with patch(
        "src.application.services.orchestrator.execution_collect.apply_recovery_hedge_to_candidates",
        return_value=[],
    ):
        orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == PAIR


def test_collect_cluster_orders_mandatory_returns_empty_when_fallback_missing():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR],
        config={"orchestrator": {"execution": {}}},
        risk_manager=SimpleNamespace(
            pending_loss={ANCHOR: 4.0},
            last_loss_symbol=ANCHOR,
            last_loss_direction="PUT",
        ),
        _active_cycle_id=12,
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
                "execute": False,
                "deploy_ok": True,
                "trade_score": 0.65,
                "val_accuracy": 0.55,
                "raw_prob": 0.58,
            },
        },
    }
    with (
        patch(
            "src.application.services.orchestrator.execution_collect.gather_cluster_candidates",
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.pick_entropy_fallback_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.build_mandatory_fallback_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.pick_absolute_mandatory_candidate",
            return_value=None,
        ),
    ):
        assert collect_cluster_orders(exec_mgr, decisions) == []


def test_resolve_mandatory_ultimate_candidate_skips_when_not_mandatory():
    exec_mgr = SimpleNamespace(
        orch=SimpleNamespace(risk_manager=SimpleNamespace(consecutive_losses=0)),
        _trade_symbols=lambda: [ANCHOR],
    )
    best, pool = resolve_mandatory_ultimate_candidate(
        exec_mgr,
        {},
        mandatory=False,
        recovery_active=False,
        last_loss=None,
        last_loss_dir=None,
        skip_symbols=frozenset(),
        min_signal=0.5,
        min_val=0.5,
        mean_reversion=True,
        low_accuracy=True,
    )
    assert best is None
    assert pool is None


def test_collect_cluster_orders_skips_entry_without_inferable_direction():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False}},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.0}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
        ),
        _active_cycle_id=3,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {
        PAIR: {"direction": None, "metrics": {"execute": True, "raw_prob": 0.55}},
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == PAIR
    assert orders[0][1] == TradeDirection.CALL


def test_collect_cluster_orders_does_not_exclude_symbol_by_loss_streak():
    rm = SimpleNamespace(
        recovery_symbol_loss_streak={ANCHOR: 5},
        dlambert_config={"recovery_max_losses_per_symbol": 2},
    )
    assert recovery_blocked_symbols(rm, {}) == frozenset()


def test_collect_cluster_orders_uses_ultimate_fallback_when_select_empty():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False}},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.45}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
        ),
        _active_cycle_id=5,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [PAIR],
    )
    pool = [(PAIR, TradeDirection.PUT, {"trade_score": 0.55, "execute": True})]
    fallback = (PAIR, TradeDirection.PUT, {"trade_score": 0.55, "dl_direction": "PUT", "exec_direction": "PUT"})
    with (
        patch(
            "src.application.services.orchestrator.execution_collect.gather_cluster_candidates",
            return_value=pool,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect._select_cluster_best",
            return_value=None,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.build_mandatory_fallback_candidate",
            return_value=fallback,
        ),
    ):
        decisions = {
            PAIR: {
                "direction": TradeDirection.PUT,
                "metrics": {"execute": True, "trade_score": 0.55, "val_accuracy": 0.55, "deploy_ok": True},
            },
        }
        orders = collect_cluster_orders(exec_mgr, decisions)
    assert orders == [fallback]


def test_collect_cluster_orders_mandatory_returns_empty_when_select_none():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={"orchestrator": {"execution": {"include_anchor_trades": False}}},
        risk_manager=SimpleNamespace(pending_loss={}, last_loss_symbol=None, last_loss_direction=None),
        _active_cycle_id=4,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {
        PAIR: {"direction": TradeDirection.CALL, "metrics": {"execute": True, "raw_prob": 0.6}},
    }
    with (
        patch(
            "src.application.services.orchestrator.execution_collect.select_mandatory_execution_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.build_mandatory_fallback_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.pick_absolute_mandatory_candidate",
            return_value=None,
        ),
    ):
        assert collect_cluster_orders(exec_mgr, decisions) == []
