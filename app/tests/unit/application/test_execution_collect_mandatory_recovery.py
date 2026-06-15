from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, LOW_SIDE_SYMBOL, PAIR


def test_collect_cluster_orders_recovery_picks_dl_put_after_call_loss():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[LOW_SIDE_SYMBOL, PAIR, ANCHOR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True}},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.45}},
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={LOW_SIDE_SYMBOL: 1.37},
            last_loss_symbol=LOW_SIDE_SYMBOL,
            last_loss_direction="CALL",
            consecutive_losses=1,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
        ),
        _active_cycle_id=2,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [LOW_SIDE_SYMBOL, PAIR, ANCHOR],
    )
    decisions = {
        ANCHOR: {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "trade_score": 0.64,
                "val_accuracy": 0.80,
                "raw_prob": 0.48,
                "deploy_ok": True,
            },
        },
        LOW_SIDE_SYMBOL: {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.49, "val_accuracy": 0.67, "deploy_ok": True},
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == ANCHOR
    assert orders[0][1] == TradeDirection.PUT


def test_collect_cluster_orders_recovery_flips_r100_put_after_call_loss():
    orch = SimpleNamespace(
        anchor="R_100",
        symbols=["R_100"],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True, "recovery_flip_direction_after_loss": True}},
            "risk_management": {"kelly": {}},
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={"R_100": 14.32},
            last_loss_symbol="R_100",
            last_loss_direction="CALL",
            consecutive_losses=1,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
        ),
        _active_cycle_id=4,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: ["R_100"],
    )
    decisions = {
        "R_100": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": True,
                "trade_score": 0.62,
                "val_accuracy": 0.63,
                "raw_prob": 0.62,
                "deploy_ok": True,
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == "R_100"
    assert orders[0][1] == TradeDirection.PUT
    assert orders[0][2].get("direction_inverted") is True


def test_collect_cluster_orders_recovery_skips_weak_signal():
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
            consecutive_losses=0,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
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
            "direction": None,
            "metrics": {
                "raw_prob": 0.40,
                "execute": False,
                "val_accuracy": 0.52,
                "trade_score": 0.48,
                "deploy_ok": True,
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][1] == TradeDirection.PUT
