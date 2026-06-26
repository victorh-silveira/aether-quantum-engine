from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.application.services.orchestrator.execution_collect import (
    _gather_cluster_candidates,
    cluster_entry_eligible,
    collect_cluster_orders,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


def test_gather_cluster_candidates_skips_unbuildable_direction():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[PAIR],
        _active_cycle_id=1,
        config={"orchestrator": {"execution": {}}},
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


def test_gather_cluster_candidates_skips_when_build_returns_none():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[PAIR],
        _active_cycle_id=1,
        config={"orchestrator": {"execution": {}}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "raw_prob": 0.82, "deploy_ok": True},
        },
    }
    with patch(
        "src.application.services.orchestrator.execution_collect.build_execution_candidate",
        return_value=None,
    ):
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


def test_collect_cluster_orders_skips_best_below_final_quality_floor():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False}},
            "deep_learning": {"min_edge_execute": 0.04},
            "risk_management": {"kelly": {"mandatory_min_trade_score": 0.68}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
            consecutive_losses=0,
        ),
        _active_cycle_id=12,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: False,
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {
        PAIR: {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": True,
                "deploy_ok": True,
                "trade_score": 0.82,
                "val_accuracy": 0.70,
                "edge": 0.12,
                "raw_prob": 0.82,
                "trend_direction": "CALL",
                "indicators": {"adx": 0.28, "hurst": 0.55, "vol_ratio": 1.1, "rsi": 0.52, "keltner": 0.55, "cmo": 0.05},
            },
        },
    }
    with patch(
        "src.application.services.orchestrator.execution_collect.passes_execution_quality",
        side_effect=[True, False],
    ):
        orders = collect_cluster_orders(exec_mgr, decisions)
    assert orders == []
    assert exec_mgr.logger.info.called


def test_collect_cluster_orders_inverts_on_grey_zone():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False}},
            "deep_learning": {"recovery_gating": {}},
            "risk_management": {"kelly": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={},
            last_loss_symbol=None,
            last_loss_direction=None,
            consecutive_losses=0,
            kelly_config={},
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
    assert len(orders) == 1
    assert orders[0][0] == PAIR
    assert orders[0][1] == TradeDirection.PUT  # Invertido de CALL para PUT


def test_cluster_entry_recovery_accepts_mandatory_weak_with_pending_loss():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "execute": False,
            "trade_score": 0.51,
            "val_accuracy": 0.59,
            "raw_prob": 0.49,
            "deploy_ok": True,
        },
    }
    assert cluster_entry_eligible(
        entry,
        mandatory=True,
        recovery_active=True,
        recovery_cfg={},
        min_signal=0.50,
        min_val=0.50,
    )


def test_collect_cluster_orders_allows_weak_fallback_in_recovery():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False}},
            "risk_management": {
                "kelly": {
                    "max_consecutive_losses": 0,
                }
            },
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 100.0},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
            consecutive_losses=0,
            risk_params={"payout_estimate": 0.95},
            kelly_config={"max_consecutive_losses": 0},
            proposal_skip_symbols=frozenset,
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
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": False,
                "trade_score": 0.20,
                "val_accuracy": 0.30,
                "raw_prob": 0.52,
                "deploy_ok": True,
            },
        },
    }
    orders = collect_cluster_orders(exec_mgr, decisions)
    assert len(orders) == 1
