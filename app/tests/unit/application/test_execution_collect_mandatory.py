from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.application.services.execution_symbols_recovery import recovery_blocked_symbols
from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.application.services.orchestrator.execution_collect_helpers import (
    mandatory_fallback_candidates,
    resolve_mandatory_ultimate_candidate,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR
from tests.unit.application.universal_regime_metrics import bear_put_metrics


def test_mandatory_fallback_candidates_returns_entropy_pick():
    exec_mgr = SimpleNamespace(
        _trade_symbols=lambda: [ANCHOR],
        orch=SimpleNamespace(risk_manager=SimpleNamespace(consecutive_losses_linear=0)),
    )
    entropy_candidate = (ANCHOR, TradeDirection.CALL, {"trade_score": 0.55})
    with patch(
        "src.application.services.orchestrator.execution_collect_helpers.pick_entropy_fallback_candidate",
        return_value=entropy_candidate,
    ):
        picks = mandatory_fallback_candidates(
            exec_mgr,
            {},
            recovery_active=False,
            last_loss_symbol=None,
            skip_symbols=frozenset(),
            min_signal=0.45,
            min_val=0.50,
        )
    assert picks == [entropy_candidate]


def test_collect_cluster_orders_mandatory_fallback_after_recovery_filter():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True, "regime_evaluator": {"enabled": True}}},
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={ANCHOR: 4.64},
            last_loss_symbol=ANCHOR,
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
            "metrics": bear_put_metrics(execute=False, trade_score=0.72, raw_prob=0.36, calibrated_prob=0.36),
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == PAIR


def test_collect_cluster_orders_mandatory_returns_empty_when_fallback_missing():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR],
        config={"orchestrator": {"execution": {"regime_evaluator": {"enabled": True}}}},
        risk_manager=SimpleNamespace(
            pending_loss={ANCHOR: 4.0},
            last_loss_symbol=ANCHOR,
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
        skip_symbols=frozenset(),
        min_signal=0.5,
        min_val=0.5,
    )
    assert best is None
    assert pool is None


def test_collect_cluster_orders_skips_entry_without_inferable_direction():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False, "regime_evaluator": {"enabled": True}}},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.0}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
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
        PAIR: {"direction": None, "metrics": bear_put_metrics(raw_prob=0.28, calibrated_prob=0.28)},
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == PAIR
    assert orders[0][1] == TradeDirection.CALL


def test_collect_cluster_orders_blocks_symbol_after_linear_loss():
    rm = SimpleNamespace(
        consecutive_losses_linear=1,
        last_loss_symbol=ANCHOR,
    )
    assert recovery_blocked_symbols(rm, {"symbol_loss_rotation_cycles": 1}) == frozenset({ANCHOR})


def test_collect_cluster_orders_uses_ultimate_fallback_when_select_empty():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False, "regime_evaluator": {"enabled": True}}},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.45}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
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
        config={"orchestrator": {"execution": {"include_anchor_trades": False, "regime_evaluator": {"enabled": True}}}},
        risk_manager=SimpleNamespace(pending_loss={}, last_loss_symbol=None),
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
