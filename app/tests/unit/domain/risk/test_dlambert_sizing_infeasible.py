import pytest

from src.domain.risk.dlambert_sizing import resolve_dlambert_stake


def test_resolve_dlambert_stake_infeasible_material_pending_returns_cap_dal():
    class RM:
        dlambert_unit = 10.0
        dlambert_config = {}
        soft_recovery_config = {
            "enabled": True,
            "amort_cycles_min": 2,
            "amort_cycles_max": 5,
            "max_safe_stake_pct": 0.025,
            "infeasible_force_explore": True,
            "material_pending_min": 0.25,
            "cover_enabled": True,
            "cover_multiple": 1.5,
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
    assert tag == "D'ALEMBERT"
    assert stake == pytest.approx(250.0)
    assert metrics.get("recovery_force_explore") is False
    assert metrics.get("recovery_infeasible") is True
    assert metrics.get("recovery_infeasible_cap_stake") is True


def test_resolve_dlambert_stake_weak_f_star_clamps_sticky_kelly_to_floor():
    class RM:
        dlambert_unit = 110.48
        dlambert_config = {}
        soft_recovery_config = {"enabled": False}
        risk_params = {"payout_estimate": 0.95}
        last_loss_stake = 0.0
        total_session_profit = 0.0
        daily_stop_win_target = 0.0

    stake, tag = resolve_dlambert_stake(
        recovery_active=False,
        bankroll=11000.0,
        kelly_base=110.48,
        dlambert_config={"dlambert_enabled": False},
        rm=RM(),
        consecutive_losses_linear=0,
        pending_total=0.0,
        payout=0.95,
        dl_metrics={},
        f_star=0.0,
    )
    assert tag == "KELLY"
    assert stake == pytest.approx(110.0)


def test_resolve_dlambert_stake_scale_force_explore_weak_f_star_clamps():
    class RM:
        dlambert_unit = 110.0
        dlambert_config = {}
        soft_recovery_config = {"enabled": True, "material_pending_min": 0.25}
        risk_params = {"payout_estimate": 0.95}
        last_loss_stake = 0.0
        total_session_profit = 0.0
        daily_stop_win_target = 0.0

    stake, tag = resolve_dlambert_stake(
        recovery_active=False,
        bankroll=11000.0,
        kelly_base=110.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=0,
        pending_total=0.0,
        payout=0.95,
        dl_metrics={"scale_force_explore": True},
        f_star=1e-12,
    )
    assert tag == "KELLY"
    assert stake == pytest.approx(110.0)


def test_resolve_dlambert_stake_force_explore_tiny_f_star_clamps():
    class RM:
        dlambert_unit = 110.0
        dlambert_config = {}
        soft_recovery_config = {
            "enabled": True,
            "material_pending_min": 0.25,
            "near_stop_win_freeze_pct": 0.70,
            "max_safe_stake_pct": 0.05,
        }
        risk_params = {"payout_estimate": 0.95}
        last_loss_stake = 20.0
        total_session_profit = 8.0
        daily_stop_win_target = 10.0

    metrics = {"neg_edge_soft": True}
    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=11000.0,
        kelly_base=110.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=1,
        pending_total=50.0,
        payout=0.95,
        dl_metrics=metrics,
        f_star=1e-12,
    )
    assert tag == "KELLY"
    assert stake == pytest.approx(110.0)
    assert metrics.get("recovery_force_explore") is True
    assert metrics.get("recovery_explore_used_cover") is False


def test_resolve_dlambert_force_explore_weak_f_star_with_cover_enabled():
    class RM:
        dlambert_unit = 110.0
        dlambert_config = {}
        soft_recovery_config = {
            "enabled": True,
            "cover_enabled": True,
            "material_pending_min": 0.25,
            "near_stop_win_freeze_pct": 0.70,
            "max_safe_stake_pct": 0.05,
        }
        risk_params = {"payout_estimate": 0.95}
        last_loss_stake = 20.0
        total_session_profit = 8.0
        daily_stop_win_target = 10.0

    metrics = {"neg_edge_soft": True}
    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=11000.0,
        kelly_base=110.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=1,
        pending_total=50.0,
        payout=0.95,
        dl_metrics=metrics,
        f_star=1e-12,
    )
    assert tag == "KELLY"
    assert stake == pytest.approx(110.0)
    assert metrics.get("recovery_force_explore") is True
    assert metrics.get("recovery_cover_disabled") is not True
