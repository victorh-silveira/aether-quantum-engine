from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.application.services.orchestrator.execution_collect_gather import gather_cluster_candidates
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ALT_SYMBOL, ANCHOR


def test_gather_cluster_candidates_keeps_side_eq_blocked_as_sizing_only():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ALT_SYMBOL],
        _active_cycle_id=1,
        config={"orchestrator": {"execution": {}}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [ALT_SYMBOL],
    )
    decisions = {
        ALT_SYMBOL: {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "raw_prob": 0.82, "deploy_ok": True},
        },
    }
    with patch(
        "src.application.services.orchestrator.execution_collect_gather.build_execution_candidate",
        return_value=(ALT_SYMBOL, TradeDirection.CALL, {"side_eq_blocked": True}),
    ):
        candidates = gather_cluster_candidates(
            exec_mgr,
            decisions,
            recovery_active=False,
            cid="C0001",
            min_signal=0.45,
            min_val=0.0,
        )
    assert len(candidates) == 1


def test_gather_cluster_candidates_skips_signal_status_skip():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ALT_SYMBOL],
        _active_cycle_id=1,
        config={"orchestrator": {"execution": {}}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [ALT_SYMBOL],
    )
    decisions = {
        ALT_SYMBOL: {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "raw_prob": 0.82, "deploy_ok": True},
        },
    }
    with patch(
        "src.application.services.orchestrator.execution_collect_gather.build_execution_candidate",
        return_value=(ALT_SYMBOL, TradeDirection.CALL, {"signal_status": "SKIP"}),
    ):
        candidates = gather_cluster_candidates(
            exec_mgr,
            decisions,
            recovery_active=False,
            cid="C0001",
            min_signal=0.45,
            min_val=0.0,
        )
    assert candidates == []


def test_gather_cluster_candidates_keeps_soft_loss_clf_in_pool():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ALT_SYMBOL],
        _active_cycle_id=1,
        config={"orchestrator": {"execution": {}}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [ALT_SYMBOL],
    )
    decisions = {
        ALT_SYMBOL: {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "raw_prob": 0.82, "deploy_ok": True},
        },
    }
    soft_metrics = {
        "signal_status": "OPEN",
        "loss_clf_soft": True,
        "execution_candidate_ready": True,
        "kelly_fraction_scale": 0.20,
    }
    with patch(
        "src.application.services.orchestrator.execution_collect_gather.build_execution_candidate",
        return_value=(ALT_SYMBOL, TradeDirection.CALL, soft_metrics),
    ):
        candidates = gather_cluster_candidates(
            exec_mgr,
            decisions,
            recovery_active=True,
            cid="C0029",
            min_signal=0.45,
            min_val=0.0,
        )
    assert len(candidates) == 1
    assert candidates[0][2]["loss_clf_soft"] is True


def test_gather_cluster_candidates_skips_ready_false_technical():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ALT_SYMBOL],
        _active_cycle_id=1,
        config={"orchestrator": {"execution": {}}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [ALT_SYMBOL],
    )
    decisions = {
        ALT_SYMBOL: {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "raw_prob": 0.82, "deploy_ok": True},
        },
    }
    blocked = {
        "gate_reason": "training",
        "execution_candidate_ready": False,
    }
    with patch(
        "src.application.services.orchestrator.execution_collect_gather.build_execution_candidate",
        return_value=(ALT_SYMBOL, TradeDirection.CALL, blocked),
    ):
        candidates = gather_cluster_candidates(
            exec_mgr,
            decisions,
            recovery_active=True,
            cid="C0029",
            min_signal=0.45,
            min_val=0.0,
        )
    assert candidates == []
    assert decisions[ALT_SYMBOL]["metrics"]["gate_reason"] == "training"
