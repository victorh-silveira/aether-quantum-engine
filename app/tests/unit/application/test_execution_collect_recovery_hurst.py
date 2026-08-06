import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.application.services.orchestrator.execution_collect_helpers import (
    recovery_hurst_blocks_collect,
    schedule_recovery_skip_counter_increment,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


def test_recovery_hurst_blocks_collect_increments_on_skip():
    orch = SimpleNamespace(state_store=MagicMock(), _recovery_skip_counter=0)
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [
        (
            "OTC_SPC",
            TradeDirection.CALL,
            {"indicators": {"hurst": 0.50}},
        )
    ]
    kelly_cfg = {"recovery_hurst_persistence_min": 0.58}
    blocked = recovery_hurst_blocks_collect(
        exec_mgr,
        candidates,
        recovery_active=True,
        consecutive_losses=2,
        kelly_cfg=kelly_cfg,
        cid="C0012",
        recovery_skip_counter=0,
    )
    assert blocked is True
    exec_mgr.logger.debug.assert_called()


@pytest.mark.asyncio
async def test_recovery_skip_counter_persisted_on_hurst_block():
    store = AsyncMock()
    store.get_string = AsyncMock(return_value="0")
    store.set_string = AsyncMock()
    orch = SimpleNamespace(state_store=store, _recovery_skip_counter=0)
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [
        (
            "OTC_SPC",
            TradeDirection.CALL,
            {"indicators": {"hurst": 0.50}},
        )
    ]
    blocked = recovery_hurst_blocks_collect(
        exec_mgr,
        candidates,
        recovery_active=True,
        consecutive_losses=2,
        kelly_cfg={"recovery_hurst_persistence_min": 0.58},
        cid="C0099",
        recovery_skip_counter=0,
    )
    assert blocked is True
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending)
    assert orch._recovery_skip_counter == 1


def test_schedule_recovery_skip_counter_increment_without_loop():
    orch = SimpleNamespace(_recovery_skip_counter=2)
    schedule_recovery_skip_counter_increment(orch)
    assert orch._recovery_skip_counter == 3


def test_recovery_hurst_blocks_collect_allows_after_decay_counter():
    orch = SimpleNamespace(state_store=None, _recovery_skip_counter=8)
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [
        (
            "OTC_SPC",
            TradeDirection.CALL,
            {"indicators": {"hurst": 0.55}},
        )
    ]
    kelly_cfg = {
        "recovery_hurst_persistence_min": 0.58,
        "recovery_hurst_decay_enabled": True,
        "recovery_hurst_decay_per_skip": 0.01,
        "recovery_hurst_decay_floor": 0.50,
    }
    blocked = recovery_hurst_blocks_collect(
        exec_mgr,
        candidates,
        recovery_active=True,
        consecutive_losses=2,
        kelly_cfg=kelly_cfg,
        cid="C0011",
        recovery_skip_counter=8,
    )
    assert blocked is False


def test_recovery_hurst_blocks_collect_log_decay_unblocks_n3_severe_drawdown():
    orch = SimpleNamespace(state_store=None, _recovery_skip_counter=4)
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [
        (
            "OTC_SPC",
            TradeDirection.CALL,
            {"indicators": {"hurst": 0.54}},
        )
    ]
    kelly_cfg = {
        "recovery_hurst_persistence_min": 0.58,
        "recovery_hurst_decay_enabled": True,
        "recovery_hurst_log_decay_coef": 0.025,
        "recovery_hurst_accel_losses_min": 3,
        "recovery_hurst_severe_drawdown_min": 150.0,
    }
    blocked = recovery_hurst_blocks_collect(
        exec_mgr,
        candidates,
        recovery_active=True,
        consecutive_losses=3,
        kelly_cfg=kelly_cfg,
        cid="C0015",
        recovery_skip_counter=4,
        session_drawdown=200.0,
    )
    assert blocked is False


def test_collect_cluster_orders_keeps_candidate_when_recovery_lacks_hurst_persistence():
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
            "src.application.services.orchestrator.execution_collect.gather_cluster_candidates",
            return_value=low_hurst,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect.mandatory_fallback_if_empty",
            side_effect=lambda _m, _d, candidates, **_: candidates,
        ),
    ):
        orders = collect_cluster_orders(exec_mgr, {PAIR: {"direction": TradeDirection.CALL, "metrics": {}}})
    assert len(orders) == 1
    assert orders[0][0] == PAIR
