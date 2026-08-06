import pytest

from src.domain.risk.consensus_stake_penalty import max_safe_stake_cap


def test_large_bankroll_max_safe_stake_uses_pct_not_abs_cap():
    soft = {
        "max_safe_stake_cap": 4.20,
        "max_safe_stake_pct": 0.05,
        "max_safe_stake_pct_linear2": 0.025,
        "max_safe_stake_pct_linear3": 0.020,
    }
    assert max_safe_stake_cap(12000.0, consecutive_losses_linear=0, soft_recovery=soft) == pytest.approx(600.0)
    assert max_safe_stake_cap(12000.0, consecutive_losses_linear=1, soft_recovery=soft) == pytest.approx(600.0)
    assert max_safe_stake_cap(12000.0, consecutive_losses_linear=2, soft_recovery=soft) == pytest.approx(300.0)
    assert max_safe_stake_cap(12000.0, consecutive_losses_linear=3, soft_recovery=soft) == pytest.approx(240.0)
