from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


def test_collect_cluster_orders_skips_when_recovery_lacks_hurst_persistence():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True}},
            "deep_learning": {"recovery_gating": {}},
            "risk_management": {
                "kelly": {
                    "recovery_hurst_persistence_min": 0.58,
                    "recovery_min_trade_score": 0.64,
                }
            },
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="PUT",
            consecutive_losses=2,
            recovery_symbol_loss_streak={},
            symbol_loss_cooldown={},
        ),
        _active_cycle_id=11,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: False,
        _trade_symbols=lambda: [PAIR],
    )
    low_hurst = [
        (
            PAIR,
            TradeDirection.CALL,
            {
                "trade_score": 0.70,
                "val_accuracy": 0.65,
                "edge": 0.10,
                "direction_margin": 0.08,
                "indicators": {"hurst": 0.50, "adx": 0.25},
            },
        )
    ]
    with (
        patch(
            "src.application.services.orchestrator.execution_collect._gather_cluster_candidates",
            return_value=low_hurst,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.mandatory_fallback_if_empty",
            side_effect=lambda _m, _d, candidates, **_: candidates,
        ),
    ):
        orders = collect_cluster_orders(exec_mgr, {PAIR: {"direction": TradeDirection.CALL, "metrics": {}}})
    assert orders == []
    exec_mgr.logger.info.assert_called()
