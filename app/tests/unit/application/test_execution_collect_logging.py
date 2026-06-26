from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.application.services.orchestrator.execution_collect_helpers import log_execution_decision
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


def test_log_execution_decision_direct():
    exec_mgr = SimpleNamespace(logger=MagicMock())
    best = (
        "R_10",
        TradeDirection.CALL,
        {
            "val_accuracy": 0.55,
            "raw_prob": 0.52,
            "indicators": {"hurst": 0.56, "adx": 31.0},
            "call_votes": 4,
            "put_votes": 2,
        },
    )
    log_execution_decision(exec_mgr, "C0001", best, [best], 0.55)
    assert exec_mgr.logger.info.called


def test_collect_cluster_orders_covers_logging():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True}},
            "deep_learning": {"min_val_accuracy": 0.50, "min_edge_execute": 0.04},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.68}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
        ),
        _active_cycle_id=5,
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
                "deploy_ok": True,
                "trade_score": 0.80,
                "val_accuracy": 0.70,
                "edge": 0.12,
                "raw_prob": 0.80,
                "trend_direction": "CALL",
                "indicators": {"hurst": 0.56, "adx": 0.28, "vol_ratio": 1.1, "rsi": 0.52, "keltner": 0.55, "cmo": 0.05},
                "call_votes": 3,
                "put_votes": 1,
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert exec_mgr.logger.info.called
