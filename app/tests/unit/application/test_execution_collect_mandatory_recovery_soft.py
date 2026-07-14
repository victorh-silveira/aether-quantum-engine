from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, HIGH_SIDE_SYMBOL, PAIR
from tests.unit.application.universal_regime_metrics import asymmetric_gate_safe_metrics


def test_collect_cluster_orders_recovery_waives_rotation_when_pool_would_empty():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[PAIR, HIGH_SIDE_SYMBOL],
        config={
            "orchestrator": {
                "execution": {
                    "include_anchor_trades": False,
                    "regime_evaluator": {"enabled": True},
                    "loss_protection": {"min_direction_margin": 0.18, "recovery_min_hurst": 0.50},
                }
            },
            "risk_management": {"kelly": {"symbol_loss_rotation_cycles": 1, "mandatory_min_trade_score": 0.45}},
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={HIGH_SIDE_SYMBOL: 38.56},
            last_loss_symbol=HIGH_SIDE_SYMBOL,
            last_loss_direction="CALL",
            consecutive_losses=1,
            consecutive_losses_linear=1,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
            proposal_skip_symbols=frozenset,
            pending_loss_total=lambda: 38.56,
        ),
        _active_cycle_id=1,
        _recovery_skip_counter=0,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [PAIR, HIGH_SIDE_SYMBOL],
    )
    decisions = {
        PAIR: {
            "direction": TradeDirection.PUT,
            "metrics": asymmetric_gate_safe_metrics(
                trade_score=0.70,
                raw_prob=0.35,
                calibrated_prob=0.35,
                direction_margin=0.30,
                indicators={"adx": 0.28, "hurst": 0.42, "vol_ratio": 1.10, "rsi": 0.55, "cmo": 0.10, "bb_width": 0.05},
            ),
        },
        HIGH_SIDE_SYMBOL: {
            "direction": TradeDirection.CALL,
            "metrics": asymmetric_gate_safe_metrics(
                trade_score=0.72,
                raw_prob=0.67,
                calibrated_prob=0.67,
                direction_margin=0.34,
            ),
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] == HIGH_SIDE_SYMBOL
    assert orders[0][1] == TradeDirection.CALL


def test_collect_cluster_orders_recovery_allows_soft_meta_zscore_reject():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[PAIR, HIGH_SIDE_SYMBOL],
        config={
            "orchestrator": {
                "execution": {
                    "include_anchor_trades": False,
                    "regime_evaluator": {"enabled": True},
                    "quality_gate": {"min_meta_payoff_zscore": 0.5, "min_direction_margin": 0.04},
                }
            },
            "risk_management": {"kelly": {"symbol_loss_rotation_cycles": 1, "mandatory_min_trade_score": 0.45}},
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={HIGH_SIDE_SYMBOL: 540.66},
            last_loss_symbol=HIGH_SIDE_SYMBOL,
            last_loss_direction="CALL",
            consecutive_losses=4,
            consecutive_losses_linear=4,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
            proposal_skip_symbols=frozenset,
            pending_loss_total=lambda: 540.66,
        ),
        _active_cycle_id=71,
        _recovery_skip_counter=0,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [PAIR, HIGH_SIDE_SYMBOL],
    )
    decisions = {
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": asymmetric_gate_safe_metrics(
                trade_score=0.51,
                raw_prob=0.51,
                calibrated_prob=0.51,
                direction_margin=0.02,
                predicted_payoff_edge=1.30,
                meta_payoff_edge_zscore=0.10,
                edge_zscore=0.10,
                edge_zscore_samples=15,
                quality_guard_reject=True,
                execution_gate_state="meta_zscore_reject",
                quality_gate_reason="[Meta Z-Score 0.10 < min 0.50]",
            ),
        },
        HIGH_SIDE_SYMBOL: {
            "direction": TradeDirection.CALL,
            "metrics": asymmetric_gate_safe_metrics(
                trade_score=0.64,
                raw_prob=0.64,
                calibrated_prob=0.64,
                direction_margin=0.08,
                predicted_payoff_edge=1.40,
                meta_payoff_edge_zscore=0.12,
                edge_zscore=0.12,
                edge_zscore_samples=15,
                quality_guard_reject=True,
                execution_gate_state="meta_zscore_reject",
                quality_gate_reason="[Meta Z-Score 0.12 < min 0.50]",
            ),
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
    assert orders[0][0] in {PAIR, HIGH_SIDE_SYMBOL}
