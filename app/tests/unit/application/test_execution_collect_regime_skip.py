from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.execution_collect import (
    _regime_skip_blocks_mandatory_cycle,
    _skip_label_for_gate,
    collect_cluster_orders,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR
from tests.unit.application.universal_regime_metrics import base_metrics


def test_skip_label_for_gate_maps_known_and_default_reasons():
    assert _skip_label_for_gate("low_conviction_neutral_skip", "X") == "LOW CONVICTION NEUTRAL SKIP"
    assert _skip_label_for_gate("micro_adx_chop_skip", "X") == "MICRO ADX CHOP SKIP"
    assert _skip_label_for_gate("micro_squeeze_breakout_skip", "X") == "MICRO SQUEEZE BREAKOUT SKIP"
    assert _skip_label_for_gate(None, "DEFAULT") == "DEFAULT"
    assert _skip_label_for_gate("unknown_reason", "DEFAULT") == "DEFAULT"


def test_collect_cluster_orders_mandatory_blocks_entropic_regime_skip():
    entropic_metrics = base_metrics(
        call_votes=3,
        put_votes=3,
        deploy_ok=True,
        execute=True,
        val_accuracy=0.60,
        edge=0.08,
        indicators={"hurst": 0.40, "adx": 0.18, "vol_ratio": 0.90, "rsi": 0.50, "cmo": 0.05},
    )
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {
                "execution": {
                    "mandatory_trade_each_cycle": True,
                    "regime_evaluator": {"enabled": True},
                },
            },
            "deep_learning": {"recovery_gating": {}, "min_val_accuracy": 0.54},
            "risk_management": {"kelly": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={ANCHOR: 10.0},
            last_loss_symbol=ANCHOR,
            last_loss_direction="CALL",
            consecutive_losses_linear=2,
            total_session_profit=-10.0,
            symbol_loss_cooldown={},
            recovery_symbol_loss_streak={},
        ),
        _active_cycle_id=9,
        _recovery_skip_counter=0,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [ANCHOR, PAIR],
    )
    decisions = {
        ANCHOR: {"direction": TradeDirection.CALL, "metrics": dict(entropic_metrics)},
        PAIR: {"direction": TradeDirection.PUT, "metrics": dict(entropic_metrics)},
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert orders == []
    logged = " ".join(
        " ".join(str(part) for part in call.args) for call in exec_mgr.logger.info.call_args_list if call.args
    )
    assert "ENTROPIC_NOISE" in logged or "LOW CONVICTION NEUTRAL SKIP" in logged


def test_regime_skip_blocks_mandatory_cycle_skips_unbuildable_symbols():
    entropic_metrics = base_metrics(
        call_votes=3,
        put_votes=3,
        deploy_ok=True,
        execute=True,
        val_accuracy=0.60,
        indicators={"hurst": 0.40, "adx": 0.18, "vol_ratio": 0.90, "rsi": 0.50, "cmo": 0.05},
    )
    orch = SimpleNamespace(
        config={
            "orchestrator": {"execution": {"regime_evaluator": {"enabled": True}}},
            "deep_learning": {"calibration": {}},
        },
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        _trade_symbols=lambda: [ANCHOR, PAIR],
    )
    decisions = {
        ANCHOR: {"direction": TradeDirection.CALL, "metrics": {"gate_reason": "training"}},
        PAIR: {"direction": TradeDirection.PUT, "metrics": dict(entropic_metrics)},
    }
    blocked, _ = _regime_skip_blocks_mandatory_cycle(exec_mgr, decisions, recovery_active=False)
    assert blocked is True


def test_collect_cluster_orders_regime_skip_ignores_unbuildable_symbol():
    trend_metrics = base_metrics(
        deploy_ok=True,
        execute=True,
        val_accuracy=0.60,
        edge=0.08,
        indicators={"adx": 0.28, "hurst": 0.58, "vol_ratio": 1.10, "rsi": 0.55, "cmo": 0.10},
    )
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {
                "execution": {
                    "mandatory_trade_each_cycle": True,
                    "regime_evaluator": {"enabled": True},
                },
            },
            "deep_learning": {"recovery_gating": {}, "min_val_accuracy": 0.54},
            "risk_management": {"kelly": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
            consecutive_losses_linear=0,
            total_session_profit=0.0,
            symbol_loss_cooldown={},
            recovery_symbol_loss_streak={},
        ),
        _active_cycle_id=11,
        _recovery_skip_counter=0,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [ANCHOR, PAIR],
    )
    decisions = {
        ANCHOR: {"direction": TradeDirection.CALL, "metrics": {"gate_reason": "training"}},
        PAIR: {"direction": TradeDirection.CALL, "metrics": dict(trend_metrics)},
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == PAIR
