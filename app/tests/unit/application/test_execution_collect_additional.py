from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.execution_collect import (
    _gather_cluster_candidates,
    collect_cluster_orders,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


def test_gather_cluster_candidates_skips_unbuildable_direction():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[PAIR],
        _active_cycle_id=1,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {PAIR: {"direction": None, "metrics": {"execute": True}}}
    candidates = _gather_cluster_candidates(
        exec_mgr,
        decisions,
        mandatory=False,
        recovery_active=False,
        recovery_cfg={},
        cid="C0001",
        min_signal=0.45,
        min_val=0.0,
    )
    assert candidates == []


def test_collect_cluster_orders_warns_on_grey_zone():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False}},
            "deep_learning": {"recovery_gating": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
            consecutive_losses=0,
        ),
        _active_cycle_id=15,
        _dl_brief_last_logged=None,
        _dl_brief_last_key=None,
    )
    logger_mock = MagicMock()
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=logger_mock,
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": False,
                "deploy_ok": True,
                "val_accuracy": 0.49,
                "conviction": 0.55,
                "raw_prob": 0.55,
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert orders == []
    assert any("zona cinza" in str(call.args[0]) for call in logger_mock.info.call_args_list)
