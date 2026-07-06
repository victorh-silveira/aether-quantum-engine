import pytest

from src.application.services.direction_loss_tracker import (
    anti_trend_lock_active,
    consecutive_direction_losses,
    direction_loss_tracker_snapshot,
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
