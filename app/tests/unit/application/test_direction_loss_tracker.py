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
    should_anti_trend_lock_flip,
)
from src.domain.models.trade import TradeDirection


@pytest.fixture(autouse=True)
def _reset_tracker():
    reset_direction_persistence_tracker()
    yield
    reset_direction_persistence_tracker()


def test_record_direction_outcome_tracks_consecutive_losses():
    record_direction_outcome("R_10", "CALL", won=False)
    record_direction_outcome("R_10", "CALL", won=False)
    assert consecutive_direction_losses("R_10", "CALL") == 2
    assert anti_trend_lock_active("R_10", TradeDirection.CALL) is True
    record_direction_outcome("R_10", "CALL", won=True)
    assert consecutive_direction_losses("R_10", "CALL") == 0


def test_direction_loss_tracker_snapshot():
    record_direction_outcome("R_10", "PUT", won=False)
    snap = direction_loss_tracker_snapshot()
    assert snap["R_10"]["PUT"] == 1


def test_consecutive_direction_losses_ignores_invalid_direction():
    assert consecutive_direction_losses("R_10", "SIDE") == 0
    record_direction_outcome("R_10", None, won=False)


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
        tracker.record_outcome("R_10", "PUT", won=False)
        tracker.record_outcome("R_10", "PUT", won=False)
        assert tracker.consecutive_losses("R_10", "PUT") == 2
        assert tracker.anti_trend_lock_active("R_10", TradeDirection.PUT) is True
        tracker.prune_obsolete_direction_losses(max_age_seconds=120.0)
    assert tracker.consecutive_losses("R_10", "PUT") == 0
    assert tracker.anti_trend_lock_active("R_10", TradeDirection.PUT) is False


def test_prune_obsolete_direction_losses_keeps_fresh_memory():
    tracker = DirectionLossTracker()
    with patch(
        "src.application.services.direction_loss_tracker._cooperative_loop_time",
        side_effect=[100.0, 100.0, 200.0],
    ):
        tracker.record_outcome("R_10", "PUT", won=False)
        tracker.record_outcome("R_10", "PUT", won=False)
        tracker.prune_obsolete_direction_losses(max_age_seconds=120.0)
    assert tracker.consecutive_losses("R_10", "PUT") == 2


def test_prune_obsolete_direction_losses_skips_entries_without_timestamp():
    tracker = DirectionLossTracker()
    tracker._loss_tracker["R_10"] = {"CALL": 0, "PUT": 2}
    with patch(
        "src.application.services.direction_loss_tracker._cooperative_loop_time",
        return_value=500.0,
    ):
        tracker.prune_obsolete_direction_losses(max_age_seconds=120.0)
    assert tracker.consecutive_losses("R_10", "PUT") == 2


@pytest.mark.asyncio
async def test_prune_obsolete_direction_losses_uses_active_event_loop_clock():
    tracker = DirectionLossTracker()
    loop = asyncio.get_running_loop()
    base = loop.time()
    with patch.object(loop, "time", side_effect=[base, base, base + 125.0]):
        tracker.record_outcome("R_10", "CALL", won=False)
        tracker.record_outcome("R_10", "CALL", won=False)
        tracker.prune_obsolete_direction_losses(max_age_seconds=120.0)
    assert tracker.consecutive_losses("R_10", "CALL") == 0


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


def test_should_anti_trend_lock_flip_cases():
    assert should_anti_trend_lock_flip(None, TradeDirection.CALL) is False
    assert should_anti_trend_lock_flip("1HZ75V", TradeDirection.PUT) is False

    record_direction_outcome("1HZ75V", "PUT", won=False)
    assert should_anti_trend_lock_flip("1HZ75V", TradeDirection.PUT, pending_loss_total=0.0) is False
    assert should_anti_trend_lock_flip("1HZ75V", TradeDirection.PUT, pending_loss_total=50.0) is True
    assert should_anti_trend_lock_flip("1HZ75V", TradeDirection.CALL, pending_loss_total=50.0) is False
    # Conviccao alta (edge >= 0.070 ou edge >= 0.035 e prob >= 0.58) ignora flip
    assert (
        should_anti_trend_lock_flip(
            "1HZ75V", TradeDirection.PUT, pending_loss_total=50.0, edge=0.08, prob=0.55, trend_direction="CALL"
        )
        is False
    )
    assert (
        should_anti_trend_lock_flip(
            "1HZ75V", TradeDirection.PUT, pending_loss_total=50.0, edge=0.04, prob=0.59, trend_direction="CALL"
        )
        is False
    )
    # Alinhamento com a tendencia macro (trend == direction e prob >= 0.58) ignora flip
    assert (
        should_anti_trend_lock_flip(
            "1HZ75V",
            TradeDirection.PUT,
            pending_loss_total=50.0,
            edge=0.02,
            prob=0.59,
            trend_direction="PUT",
        )
        is False
    )
    # Tendencia contraria nao ignora flip
    assert (
        should_anti_trend_lock_flip(
            "1HZ75V",
            TradeDirection.PUT,
            pending_loss_total=50.0,
            edge=0.02,
            prob=0.59,
            trend_direction="CALL",
        )
        is True
    )

    record_direction_outcome("1HZ75V", "PUT", won=False)
    assert should_anti_trend_lock_flip("1HZ75V", TradeDirection.PUT, pending_loss_total=0.0) is True
    assert (
        should_anti_trend_lock_flip("1HZ75V", TradeDirection.PUT, pending_loss_total=50.0, edge=0.03, prob=0.56) is True
    )


def test_record_direction_outcome_resets_opposite_direction():
    record_direction_outcome("1HZ75V", "PUT", won=False)
    assert consecutive_direction_losses("1HZ75V", "PUT") == 1
    record_direction_outcome("1HZ75V", "CALL", won=False)
    assert consecutive_direction_losses("1HZ75V", "CALL") == 1
    assert consecutive_direction_losses("1HZ75V", "PUT") == 0
