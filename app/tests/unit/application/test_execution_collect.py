from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.application.services.orchestrator.execution_collect_helpers import mandatory_fallback_if_empty
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR
from tests.unit.application.universal_regime_metrics import asymmetric_gate_safe_metrics, bear_put_metrics


def test_collect_cluster_orders_recovery_executes_best_available_signal():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True, "regime_evaluator": {"enabled": True}}},
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
            "metrics": asymmetric_gate_safe_metrics(),
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == ANCHOR
    assert orders[0][1] == TradeDirection.CALL


def test_collect_cluster_orders_includes_recovery_candidate_with_raw_prob():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False, "regime_evaluator": {"enabled": True}}},
            "deep_learning": {"min_edge_execute": 0.04},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.68, "recovery_min_trade_score": 0.64}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
            consecutive_losses=0,
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
        PAIR: {
            "direction": TradeDirection.PUT,
            "metrics": bear_put_metrics(
                raw_prob=0.28,
                calibrated_prob=0.28,
                trade_score=0.82,
                conviction=0.82,
                execute=True,
            ),
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == PAIR
    assert orders[0][1] == TradeDirection.PUT


def test_collect_cluster_orders_bolts_weak_signal_continuously():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False, "regime_evaluator": {"enabled": True}}},
            "deep_learning": {"min_edge_execute": 0.04},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.68, "recovery_min_trade_score": 0.64}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
            consecutive_losses=0,
            consecutive_losses_linear=1,
            pending_loss_total=lambda: 10.0,
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
        PAIR: {
            "direction": TradeDirection.PUT,
            "metrics": {
                "raw_prob": 0.4,
                "calibrated_prob": 0.4,
                "execute": False,
                "deploy_ok": True,
                "val_accuracy": 0.55,
                "quality_guard_reject": True,
                "execution_gate_state": "meta_zscore_reject",
                "quality_gate_reason": "[Meta Z-Score -1.50 < min 0.50]",
                "meta_payoff_edge_zscore": -1.50,
                "edge_zscore": -1.50,
                "predicted_payoff_edge": -0.80,
                "edge_expectancy": "LOSS_EXPECTED",
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1

    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False, "regime_evaluator": {"enabled": True}}},
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
        PAIR: {
            "direction": TradeDirection.PUT,
            "metrics": bear_put_metrics(raw_prob=0.28, execute=True),
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1


def test_collect_cluster_orders_filters_proposal_skip_symbols():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True, "regime_evaluator": {"enabled": True}}},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.0}},
            "deep_learning": {},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
            proposal_skip_symbols=lambda: frozenset({ANCHOR}),
        ),
        _active_cycle_id=11,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: False,
        _trade_symbols=lambda: [ANCHOR, PAIR],
    )
    decisions = {
        ANCHOR: {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": True,
                "trade_score": 0.70,
                "raw_prob": 0.75,
                "calibrated_prob": 0.75,
                "deploy_ok": True,
                "predicted_payoff_edge": 0.06,
                "meta_classifier_applied": True,
            },
        },
        PAIR: {"direction": TradeDirection.PUT, "metrics": bear_put_metrics(execute=True, trade_score=0.70)},
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == PAIR


def test_mandatory_fallback_if_empty_returns_early_when_not_mandatory():
    exec_mgr = SimpleNamespace(
        orch=SimpleNamespace(config={}),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: ["RDBULL"],
    )
    kept = mandatory_fallback_if_empty(
        exec_mgr,
        {},
        [],
        mandatory=False,
        recovery_active=False,
        last_loss=None,
        last_loss_dir=None,
        skip_symbols=frozenset(),
        min_signal=0.5,
        min_val=0.5,
    )
    assert kept == []
