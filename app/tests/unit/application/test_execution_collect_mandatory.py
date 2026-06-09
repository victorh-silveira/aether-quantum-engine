from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.application.services.orchestrator.execution_collect import collect_cluster_orders
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
            "metrics": {"execute": False, "trade_score": 0.65, "val_accuracy": 0.50, "raw_prob": 0.47},
        },
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.40, "val_accuracy": 0.47, "raw_prob": 0.52},
        },
    }
    with patch(
        "src.application.services.orchestrator.execution_collect.apply_recovery_hedge_to_candidates",
        return_value=[],
    ):
        orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][1] == TradeDirection.PUT


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
            "src.application.services.orchestrator.execution_collect.apply_recovery_hedge_to_candidates",
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.build_mandatory_fallback_candidate",
            return_value=None,
        ),
    ):
        assert collect_cluster_orders(exec_mgr, decisions) == []
