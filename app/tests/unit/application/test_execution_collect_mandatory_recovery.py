from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, LOW_SIDE_SYMBOL, PAIR
from tests.unit.application.universal_regime_metrics import asymmetric_gate_safe_metrics, bear_put_metrics


def test_collect_cluster_orders_recovery_picks_dl_put_after_call_loss():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[LOW_SIDE_SYMBOL, PAIR, ANCHOR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True, "regime_evaluator": {"enabled": True}}},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.45}},
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={LOW_SIDE_SYMBOL: 1.37},
            last_loss_symbol=LOW_SIDE_SYMBOL,
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
            "direction": TradeDirection.CALL,
            "metrics": asymmetric_gate_safe_metrics(
                execute=False,
                trade_score=0.49,
                raw_prob=0.51,
                calibrated_prob=0.51,
            ),
        },
        LOW_SIDE_SYMBOL: {
            "direction": TradeDirection.PUT,
            "metrics": bear_put_metrics(
                trade_score=0.72,
                raw_prob=0.28,
                calibrated_prob=0.28,
                trend_direction="PUT",
            ),
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == LOW_SIDE_SYMBOL
    assert orders[0][1] == TradeDirection.PUT


def test_collect_cluster_orders_recovery_keeps_dl_direction_after_call_loss():
    orch = SimpleNamespace(
        anchor="R_10",
        symbols=["R_10"],
        config={
            "orchestrator": {
                "execution": {
                    "include_anchor_trades": True,
                    "regime_evaluator": {"enabled": True},
                }
            },
            "risk_management": {"kelly": {}},
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={"R_10": 14.32},
            last_loss_symbol="R_10",
            consecutive_losses=1,
            consecutive_losses_linear=1,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
            proposal_skip_symbols=frozenset,
        ),
        _active_cycle_id=4,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: ["R_10"],
    )
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": asymmetric_gate_safe_metrics(trade_score=0.72, raw_prob=0.54),
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == "R_10"
    assert orders[0][1] == TradeDirection.CALL


def test_collect_cluster_orders_recovery_bolts_hard_meta_reject():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False, "regime_evaluator": {"enabled": True}}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            consecutive_losses=0,
            consecutive_losses_linear=1,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
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
                "raw_prob": 0.40,
                "calibrated_prob": 0.40,
                "execute": False,
                "val_accuracy": 0.52,
                "trade_score": 0.48,
                "deploy_ok": True,
                "quality_guard_reject": True,
                "execution_gate_state": "meta_zscore_reject",
                "quality_gate_reason": "[Meta Z-Score -1.20 < min 0.50]",
                "meta_payoff_edge_zscore": -1.20,
                "edge_zscore": -1.20,
                "predicted_payoff_edge": -0.55,
                "edge_expectancy": "LOSS_EXPECTED",
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert orders == []


def test_collect_cluster_orders_recovery_allows_soft_tcn_entropy_fallback():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False, "regime_evaluator": {"enabled": True}}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            consecutive_losses=0,
            consecutive_losses_linear=1,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
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
            "direction": None,
            "metrics": {
                "raw_prob": 0.40,
                "execute": False,
                "val_accuracy": 0.52,
                "trade_score": 0.48,
                "deploy_ok": True,
                "quality_guard_reject": True,
                "quality_gate_reason": "[TCN Margin 0.02 < min 0.04]",
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == PAIR
    assert orders[0][1] == TradeDirection.PUT
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR],
        config={"orchestrator": {"execution": {"regime_evaluator": {"enabled": True}}}},
        risk_manager=SimpleNamespace(
            pending_loss={ANCHOR: 4.0},
            last_loss_symbol=ANCHOR,
            consecutive_losses=1,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
        ),
        _active_cycle_id=15,
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
                "trade_score": 0.40,
                "val_accuracy": 0.45,
                "raw_prob": 0.40,
                "calibrated_prob": 0.40,
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == ANCHOR
