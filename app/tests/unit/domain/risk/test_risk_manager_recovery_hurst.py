import pytest

from src.domain.risk.risk_manager import RiskManager


def test_recovery_signal_floor_delegates_hurst(kelly_config):
    rm = RiskManager(kelly_config)
    rm.consecutive_losses_linear = 2
    low = rm.recovery_signal_floor(0.45)
    high = rm.recovery_signal_floor(0.60)
    assert high <= low


@pytest.mark.parametrize(
    ("losses", "expected_min"),
    [(1, 0.52), (3, 0.56), (4, 0.58), (5, 0.58)],
)
def test_recovery_signal_floor_streak_minimums(kelly_config, losses, expected_min):
    rm = RiskManager(kelly_config)
    rm.consecutive_losses = losses
    assert rm.recovery_signal_floor(0.60) >= expected_min
