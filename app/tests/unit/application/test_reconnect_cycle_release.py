import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.application.services.orchestrator.reconnect_cycle_release import (
    _pending_loss_total,
    release_trading_cycle_after_reconnect,
    resolve_post_reconnect_warm_up_delay_seconds,
    schedule_post_reconnect_warm_up_barrier,
)


def test_pending_loss_total_without_risk_manager():
    assert _pending_loss_total(None) == 0.0


def test_pending_loss_total_from_dict():
    rm = SimpleNamespace(pending_loss={"RDBULL": 3.5})
    assert _pending_loss_total(rm) == pytest.approx(3.5)


def test_pending_loss_total_from_dict_via_resolve_warm_up():
    orch = SimpleNamespace(
        config={"orchestrator": {"stream_warm_up_delay_seconds": 45}},
        risk_manager=SimpleNamespace(pending_loss={"RDBULL": 2.0, "RDBEAR": 2.14}),
    )
    assert resolve_post_reconnect_warm_up_delay_seconds(orch) == pytest.approx(5.0)


def test_resolve_post_reconnect_warm_up_delay_seconds_shortens_with_pending():
    orch = SimpleNamespace(
        config={"orchestrator": {"stream_warm_up_delay_seconds": 45}},
        risk_manager=SimpleNamespace(pending_loss={"RDBULL": 12.0}),
    )
    assert resolve_post_reconnect_warm_up_delay_seconds(orch) == pytest.approx(5.0)


def test_resolve_post_reconnect_warm_up_delay_seconds_uses_config_without_pending():
    orch = SimpleNamespace(
        config={"orchestrator": {"stream_warm_up_delay_seconds": 30}},
        risk_manager=SimpleNamespace(pending_loss={}),
    )
    assert resolve_post_reconnect_warm_up_delay_seconds(orch) == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_schedule_post_reconnect_warm_up_barrier_sets_deadline():
    orch = SimpleNamespace(
        config={"orchestrator": {"stream_warm_up_delay_seconds": 45}},
        risk_manager=SimpleNamespace(pending_loss={}),
        logger=MagicMock(),
    )
    loop = asyncio.get_running_loop()
    base = loop.time()
    delay = schedule_post_reconnect_warm_up_barrier(orch)
    assert delay == pytest.approx(45.0)
    assert orch._stream_warmed_up_at - base == pytest.approx(45.0, abs=0.05)


@pytest.mark.asyncio
async def test_release_trading_cycle_after_reconnect_clears_signature_and_epoch():
    orch = SimpleNamespace(
        config={"orchestrator": {"stream_warm_up_delay_seconds": 45}},
        risk_manager=SimpleNamespace(
            pending_loss={"RDBULL": 4.14},
            consecutive_losses_linear=2,
            pending_loss_total=lambda: 4.14,
        ),
        last_data_signature="sig-old",
        _signature_invalidation_logged_key="sig-old",
        _last_processed_epoch=123,
        _quality_guard_logged_cycle_id=9,
        logger=MagicMock(),
    )
    release_trading_cycle_after_reconnect(orch)
    assert orch.last_data_signature == ""
    assert orch._last_processed_epoch == 0
    assert orch._quality_guard_logged_cycle_id == -1
    orch.logger.info.assert_called_once()
