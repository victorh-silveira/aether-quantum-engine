import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.application.services.direction_loss_tracker import (
    DirectionLossTracker,
    _cooperative_loop_time,
    anti_trend_lock_active,
    consecutive_direction_losses,
    direction_loss_tracker_snapshot,
    get_direction_loss_tracker,
    record_direction_outcome,
    reset_direction_persistence_tracker,
)
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_tracker():
    reset_direction_persistence_tracker()
    yield
    reset_direction_persistence_tracker()


def test_record_direction_outcome_tracks_consecutive_losses():
    record_direction_outcome("RDBULL", "CALL", won=False)
    record_direction_outcome("RDBULL", "CALL", won=False)
    assert consecutive_direction_losses("RDBULL", "CALL") == 2
    assert anti_trend_lock_active("RDBULL", TradeDirection.CALL) is True
    record_direction_outcome("RDBULL", "CALL", won=True)
    assert consecutive_direction_losses("RDBULL", "CALL") == 0


def test_direction_loss_tracker_snapshot():
    record_direction_outcome("RDBEAR", "PUT", won=False)
    snap = direction_loss_tracker_snapshot()
    assert snap["RDBEAR"]["PUT"] == 1


def test_consecutive_direction_losses_ignores_invalid_direction():
    assert consecutive_direction_losses("RDBULL", "SIDE") == 0
    record_direction_outcome("RDBULL", None, won=False)


def test_get_direction_loss_tracker_returns_singleton():
    first = get_direction_loss_tracker()
    second = get_direction_loss_tracker()
    assert first is second


def test_prune_obsolete_direction_losses_expires_stale_memory():
    tracker = DirectionLossTracker()
    with patch(
        "src.application.services.direction_loss_tracker._cooperative_loop_time",
        side_effect=[100.0, 100.0, 225.0],
    ):
        tracker.record_outcome("RDBEAR", "PUT", won=False)
        tracker.record_outcome("RDBEAR", "PUT", won=False)
        assert tracker.consecutive_losses("RDBEAR", "PUT") == 2
        assert tracker.anti_trend_lock_active("RDBEAR", TradeDirection.PUT) is True
        tracker.prune_obsolete_direction_losses(max_age_seconds=120.0)
    assert tracker.consecutive_losses("RDBEAR", "PUT") == 0
    assert tracker.anti_trend_lock_active("RDBEAR", TradeDirection.PUT) is False


def test_prune_obsolete_direction_losses_keeps_fresh_memory():
    tracker = DirectionLossTracker()
    with patch(
        "src.application.services.direction_loss_tracker._cooperative_loop_time",
        side_effect=[100.0, 100.0, 200.0],
    ):
        tracker.record_outcome("RDBEAR", "PUT", won=False)
        tracker.record_outcome("RDBEAR", "PUT", won=False)
        tracker.prune_obsolete_direction_losses(max_age_seconds=120.0)
    assert tracker.consecutive_losses("RDBEAR", "PUT") == 2


def test_prune_obsolete_direction_losses_skips_entries_without_timestamp():
    tracker = DirectionLossTracker()
    tracker._loss_tracker["RDBEAR"] = {"CALL": 0, "PUT": 2}
    with patch(
        "src.application.services.direction_loss_tracker._cooperative_loop_time",
        return_value=500.0,
    ):
        tracker.prune_obsolete_direction_losses(max_age_seconds=120.0)
    assert tracker.consecutive_losses("RDBEAR", "PUT") == 2


@pytest.mark.asyncio
async def test_prune_obsolete_direction_losses_uses_active_event_loop_clock():
    tracker = DirectionLossTracker()
    loop = asyncio.get_running_loop()
    base = loop.time()
    with patch.object(loop, "time", side_effect=[base, base, base + 125.0]):
        tracker.record_outcome("RDBULL", "CALL", won=False)
        tracker.record_outcome("RDBULL", "CALL", won=False)
        tracker.prune_obsolete_direction_losses(max_age_seconds=120.0)
    assert tracker.consecutive_losses("RDBULL", "CALL") == 0


def test_cooperative_loop_time_falls_back_to_monotonic_without_event_loop():
    with (
        patch("asyncio.get_running_loop", side_effect=RuntimeError),
        patch(
            "asyncio.get_event_loop",
            side_effect=RuntimeError,
        ),
        patch("src.application.services.direction_loss_tracker.time.monotonic", return_value=321.0),
    ):
        assert _cooperative_loop_time() == 321.0


def test_cooperative_loop_time_uses_idle_event_loop_when_available():
    mock_loop = MagicMock()
    mock_loop.is_closed.return_value = False
    mock_loop.time.return_value = 77.5
    with (
        patch("asyncio.get_running_loop", side_effect=RuntimeError),
        patch(
            "asyncio.get_event_loop",
            return_value=mock_loop,
        ),
    ):
        assert _cooperative_loop_time() == 77.5


def test_cooperative_loop_time_skips_closed_event_loop():
    mock_loop = MagicMock()
    mock_loop.is_closed.return_value = True
    with (
        patch("asyncio.get_running_loop", side_effect=RuntimeError),
        patch(
            "asyncio.get_event_loop",
            return_value=mock_loop,
        ),
        patch("src.application.services.direction_loss_tracker.time.monotonic", return_value=88.0),
    ):
        assert _cooperative_loop_time() == 88.0
