from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.application.services.orchestrator.execution_collect_gather import (
    _sync_entry_metrics,
    gather_cluster_candidates,
)
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR
from tests.unit.application.universal_regime_metrics import bear_put_metrics


FREEZE_YIELD_MODULE = "src.application.services.orchestrator.regime_freeze_yield"
TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"


def _frozen_decisions():
    return {
        "RDBULL": {
            "metrics": {
                "signal_status": SIGNAL_SUSPENDED,
                "execute": True,
                "trade_score": 0.70,
                "raw_prob": 0.70,
            }
        },
        "RDBEAR": {
            "metrics": {
                "signal_status": SIGNAL_SUSPENDED,
                "execute": True,
                "trade_score": 0.65,
                "raw_prob": 0.35,
            }
        },
    }


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


def test_gather_aborts_when_freeze_set_after_first_symbol_built(monkeypatch):
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        _active_cycle_id=4,
        config={"orchestrator": {"execution": {}}, "deep_learning": {}, "infra": {}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [PAIR, ANCHOR],
    )
    pair_metrics = bear_put_metrics(
        execute=True,
        trade_score=0.88,
        raw_prob=0.12,
        calibrated_prob=0.12,
    )
    decisions = {
        PAIR: {"direction": TradeDirection.PUT, "metrics": dict(pair_metrics)},
        ANCHOR: {
            "direction": TradeDirection.CALL,
            "metrics": {"raw_prob": 0.70, "execute": True, "deploy_ok": True},
        },
    }

    def _build(symbol, entry, **kwargs):
        if symbol == ANCHOR:
            metrics = entry["metrics"]
            metrics["signal_status"] = SIGNAL_SUSPENDED
            metrics["regime_guard_action"] = "FREEZE: SKIP CYCLE"
            return None
        return symbol, TradeDirection.PUT, dict(entry["metrics"])

    monkeypatch.setattr(
        "src.application.services.orchestrator.execution_collect_gather.build_execution_candidate",
        _build,
    )
    candidates = gather_cluster_candidates(
        exec_mgr,
        decisions,
        recovery_active=False,
        recovery_cfg={},
        cid="C0004",
        min_signal=0.45,
        min_val=0.0,
    )
    assert candidates == []
    assert decisions[PAIR]["metrics"]["signal_status"] == SIGNAL_SUSPENDED
    assert decisions[ANCHOR]["metrics"]["signal_status"] == SIGNAL_SUSPENDED


def test_sync_entry_metrics_creates_metrics_when_missing():
    entry: dict = {}
    metrics = {"signal_status": SIGNAL_SUSPENDED, "regime_guard_action": "FREEZE: SKIP CYCLE"}
    _sync_entry_metrics(entry, metrics)
    assert entry["metrics"]["signal_status"] == SIGNAL_SUSPENDED


def test_gather_aborts_after_candidate_when_freeze_propagates_mid_loop(monkeypatch):
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[PAIR],
        _active_cycle_id=5,
        config={"orchestrator": {"execution": {}}, "deep_learning": {}, "infra": {}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [PAIR],
    )
    pair_metrics = bear_put_metrics(
        execute=True,
        trade_score=0.88,
        raw_prob=0.12,
        calibrated_prob=0.12,
    )
    decisions = {PAIR: {"direction": TradeDirection.PUT, "metrics": dict(pair_metrics)}}

    def _penalty(metrics, **kwargs):
        metrics["signal_status"] = SIGNAL_SUSPENDED
        metrics["regime_guard_action"] = "FREEZE: SKIP CYCLE"
        return 1.0

    monkeypatch.setattr(
        "src.application.services.orchestrator.execution_collect_gather.apply_quality_penalty_to_metrics",
        _penalty,
    )
    monkeypatch.setattr(
        "src.application.services.orchestrator.execution_collect_gather.build_execution_candidate",
        lambda symbol, entry, **kwargs: (symbol, TradeDirection.PUT, dict(entry["metrics"])),
    )
    candidates = gather_cluster_candidates(
        exec_mgr,
        decisions,
        recovery_active=False,
        recovery_cfg={},
        cid="C0005",
        min_signal=0.45,
        min_val=0.0,
    )
    assert candidates == []
    assert decisions[PAIR]["metrics"]["signal_status"] == SIGNAL_SUSPENDED


@pytest.mark.asyncio
async def test_run_trading_cycle_freeze_skips_execute_cluster(orch_ready):
    orch = orch_ready
    orch._last_epoch = 120
    orch._last_cluster_cycle_end = 0.0
    orch._dl_fast_cycle = True
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch.executor.execute_cluster = AsyncMock()

    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=_frozen_decisions(),
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.refresh_correlation_cache", new_callable=AsyncMock),
        patch(f"{FREEZE_YIELD_MODULE}._yield_freeze_delay", new_callable=AsyncMock),
    ):
        await run_trading_cycle_if_ready(orch)

    orch.executor.execute_cluster.assert_not_called()
