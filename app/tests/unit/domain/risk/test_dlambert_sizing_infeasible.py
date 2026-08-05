import pytest

from src.domain.risk.dlambert_sizing import resolve_dlambert_stake


def test_resolve_dlambert_stake_infeasible_force_explore_returns_kelly():
    class RM:
        dlambert_unit = 10.0
        dlambert_config = {}
        soft_recovery_config = {
            "enabled": True,
            "amort_cycles_min": 2,
            "amort_cycles_max": 5,
            "max_safe_stake_pct": 0.025,
            "infeasible_force_explore": True,
        }
        risk_params = {"payout_estimate": 0.95}
        last_loss_stake = 20.0
        total_session_profit = 0.0
        daily_stop_win_target = 0.0

    metrics: dict = {}
    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=12.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=2,
        pending_total=800.0,
        payout=0.95,
        dl_metrics=metrics,
        f_star=0.01,
    )
    assert tag == "KELLY"
    assert stake == pytest.approx(12.0)
    assert metrics.get("recovery_force_explore") is True
    assert metrics.get("recovery_infeasible") is True
