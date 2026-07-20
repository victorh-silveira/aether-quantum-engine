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
from tests.market_symbols import ALT_SYMBOL, ANCHOR
from tests.unit.application.universal_regime_metrics import bear_put_metrics


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"


def _frozen_decisions():
    return {
        "R_10": {
            "metrics": {
                "signal_status": SIGNAL_SUSPENDED,
                "execute": True,
                "trade_score": 0.70,
                "raw_prob": 0.70,
            }
        },
        "R_50": {
            "metrics": {
                "signal_status": SIGNAL_SUSPENDED,
                "execute": True,
                "trade_score": 0.65,
                "raw_prob": 0.35,
            }
        },
    }


def test_gather_cluster_continues_when_anchor_frozen():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, ALT_SYMBOL],
        _active_cycle_id=3,
        config={"orchestrator": {"execution": {}}, "deep_learning": {}, "infra": {}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [ANCHOR, ALT_SYMBOL],
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
        ALT_SYMBOL: {
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
    assert len(candidates) >= 1
    assert any(item[0] == ALT_SYMBOL for item in candidates)


def test_collect_cluster_orders_continues_when_one_symbol_frozen():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, ALT_SYMBOL],
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
            consecutive_losses_linear=0,
            kelly_config={},
            proposal_skip_symbols=frozenset,
        ),
        _active_cycle_id=6,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: [ANCHOR, ALT_SYMBOL],
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
        ALT_SYMBOL: {
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
    assert len(orders) == 1
    assert orders[0][0] == ALT_SYMBOL


def test_gather_continues_when_freeze_set_after_first_symbol_built(monkeypatch):
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, ALT_SYMBOL],
        _active_cycle_id=4,
        config={"orchestrator": {"execution": {}}, "deep_learning": {}, "infra": {}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [ALT_SYMBOL, ANCHOR],
    )
    pair_metrics = bear_put_metrics(
        execute=True,
        trade_score=0.88,
        raw_prob=0.12,
        calibrated_prob=0.12,
    )
    decisions = {
        ALT_SYMBOL: {"direction": TradeDirection.PUT, "metrics": dict(pair_metrics)},
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
    assert len(candidates) == 1
    assert candidates[0][0] == ALT_SYMBOL


def test_sync_entry_metrics_creates_metrics_when_missing():
    entry: dict = {}
    metrics = {"signal_status": SIGNAL_SUSPENDED, "regime_guard_action": "FREEZE: SKIP CYCLE"}
    _sync_entry_metrics(entry, metrics)
    assert entry["metrics"]["signal_status"] == SIGNAL_SUSPENDED


def test_gather_keeps_candidate_when_freeze_metadata_present(monkeypatch):
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ALT_SYMBOL],
        _active_cycle_id=5,
        config={"orchestrator": {"execution": {}}, "deep_learning": {}, "infra": {}},
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [ALT_SYMBOL],
    )
    pair_metrics = bear_put_metrics(
        execute=True,
        trade_score=0.88,
        raw_prob=0.12,
        calibrated_prob=0.12,
    )
    decisions = {ALT_SYMBOL: {"direction": TradeDirection.PUT, "metrics": dict(pair_metrics)}}

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
    assert len(candidates) == 1
    assert candidates[0][0] == ALT_SYMBOL


@pytest.mark.asyncio
async def test_run_trading_cycle_executes_cluster_despite_signal_suspended(orch_ready):
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
    ):
        await run_trading_cycle_if_ready(orch)

    orch.executor.execute_cluster.assert_awaited_once()
