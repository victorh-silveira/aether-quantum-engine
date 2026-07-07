from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_mandatory_pick import (
    _recovery_hedge_pick,
    pick_best_mandatory_candidate,
)
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


def test_collect_cluster_orders_mandatory_keeps_weak_recovery_candidate():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR],
        config={"orchestrator": {"execution": {"regime_evaluator": {"enabled": True}}}},
        risk_manager=SimpleNamespace(
            pending_loss={ANCHOR: 4.0},
            last_loss_symbol=ANCHOR,
            last_loss_direction="PUT",
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
                "raw_prob": 0.51,
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 0


def test_pick_best_mandatory_returns_hedge_when_quality_ok():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "trade_score": 0.62,
                "raw_prob": 0.62,
                "val_accuracy": 0.60,
                "deploy_ok": True,
            },
        },
    }
    picked = pick_best_mandatory_candidate(
        ["RDBEAR", "RDBULL"],
        decisions,
        recovery_active=True,
        last_loss_symbol="RDBEAR",
        last_loss_direction="CALL",
        min_signal=0.45,
        min_val=0.50,
    )
    assert picked is not None
    assert picked[0] == "RDBULL"


def test_pick_best_mandatory_skips_hedge_when_peer_blocked():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.60, "raw_prob": 0.62, "deploy_ok": True, "val_accuracy": 0.60},
        },
    }
    picked = pick_best_mandatory_candidate(
        ["RDBEAR", "RDBULL"],
        decisions,
        recovery_active=True,
        last_loss_symbol="RDBEAR",
        last_loss_direction="CALL",
        skip_symbols=frozenset({"RDBULL"}),
        min_signal=0.45,
        min_val=0.50,
    )
    assert picked is not None
    assert picked[0] == "RDBULL"


def test_recovery_hedge_pick_returns_forced_candidate():
    decisions = {
        "RDBULL": {
            "direction": TradeDirection.CALL,
            "metrics": {"raw_prob": 0.62, "trade_score": 0.62, "deploy_ok": True},
        },
    }
    hedge = _recovery_hedge_pick(
        decisions,
        last_loss_symbol="RDBEAR",
        last_loss_direction="CALL",
        skip_symbols=frozenset(),
    )
    assert hedge is not None
    assert hedge[0] == "RDBULL"


def test_resolve_weak_without_ctx_keeps_dl_side():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "raw_prob": 0.42,
            "trade_score": 0.58,
            "val_accuracy": 0.60,
            "deploy_ok": True,
            "trend_direction": "PUT",
            "indicators": {
                "hurst": 0.55,
                "adx": 0.30,
                "vol_ratio": 1.10,
                "rsi": 0.50,
                "keltner": 0.50,
                "cmo": 0.0,
            },
        },
    }
    result = resolve_execution_direction(entry, symbol="RDBEAR")
    assert result is not None
    assert result[0] == TradeDirection.PUT
