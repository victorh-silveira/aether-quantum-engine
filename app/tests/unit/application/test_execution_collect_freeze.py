from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.application.services.orchestrator.execution_collect_gather import gather_cluster_candidates
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR
from tests.unit.application.universal_regime_metrics import bear_put_metrics


def test_gather_cluster_freeze_on_anchor_aborts_entire_cluster():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        _active_cycle_id=3,
        config={"orchestrator": {"execution": {}}, "deep_learning": {}, "infra": {}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [ANCHOR, PAIR],
    )
    decisions = {
        ANCHOR: {
            "direction": TradeDirection.CALL,
            "metrics": {
                "signal_status": SIGNAL_SUSPENDED,
                "regime_guard_action": "FREEZE: SKIP CYCLE",
                "raw_prob": 0.70,
                "execute": True,
            },
        },
        PAIR: {
            "direction": TradeDirection.PUT,
            "metrics": bear_put_metrics(
                execute=True,
                trade_score=0.85,
                raw_prob=0.15,
                calibrated_prob=0.15,
            ),
        },
    }
    candidates = gather_cluster_candidates(
        exec_mgr,
        decisions,
        recovery_active=False,
        recovery_cfg={},
        cid="C0003",
        min_signal=0.45,
        min_val=0.0,
    )
    assert candidates == []
    assert decisions[PAIR]["metrics"]["signal_status"] == SIGNAL_SUSPENDED


def test_collect_cluster_orders_aborts_pair_when_freeze_on_one_symbol():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": True}},
            "deep_learning": {"recovery_gating": {}},
            "risk_management": {"kelly": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
            consecutive_losses=0,
            kelly_config={},
            proposal_skip_symbols=frozenset,
        ),
        _active_cycle_id=6,
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
                "signal_status": SIGNAL_SUSPENDED,
                "regime_guard_action": "FREEZE: SKIP CYCLE",
                "raw_prob": 0.72,
                "execute": True,
                "deploy_ok": True,
            },
        },
        PAIR: {
            "direction": TradeDirection.PUT,
            "metrics": bear_put_metrics(
                execute=True,
                trade_score=0.88,
                raw_prob=0.12,
                calibrated_prob=0.12,
            ),
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert orders == []
    assert decisions[ANCHOR]["metrics"]["signal_status"] == SIGNAL_SUSPENDED
    assert decisions[PAIR]["metrics"]["signal_status"] == SIGNAL_SUSPENDED
