"""Testes de Kelly bayesiano, modos Explore/Recover e recovery_infeasible."""

import pytest

from src.domain.risk.bayesian_win_rate import bayesian_win_rate
from src.domain.risk.consensus_stake_penalty import apply_soft_recovery_stake, max_safe_stake_cap
from src.domain.risk.risk_manager import RiskManager
from src.domain.risk.soft_recovery_policy import is_recovery_infeasible
from src.domain.risk.stake_sizing import resolve_stake_regime


def test_bayesian_win_rate_blends_live_when_n_ready():
    p = bayesian_win_rate(
        0.70,
        rolling_wr=0.90,
        rolling_n=10,
        metrics={"live_n": 32, "live_wr": 0.40, "live_brier": 0.18, "live_ece": 0.05},
        dynamic_min_samples=6,
    )
    assert p < 0.70
    assert p == pytest.approx(0.55, abs=0.08)


def test_bayesian_win_rate_shrinks_on_high_brier():
    healthy = bayesian_win_rate(
        0.65,
        metrics={"live_n": 40, "live_wr": 0.60, "live_brier": 0.15, "live_ece": 0.04},
    )
    noisy = bayesian_win_rate(
        0.65,
        metrics={"live_n": 40, "live_wr": 0.60, "live_brier": 0.30, "live_ece": 0.04},
    )
    assert noisy < healthy


def test_bayesian_win_rate_falls_back_to_rolling_blend():
    p = bayesian_win_rate(
        0.60,
        rolling_wr=1.0,
        rolling_n=10,
        metrics={"live_n": 5},
        dynamic_min_samples=5,
    )
    assert p > 0.60
    assert p < 0.72


def test_effective_win_rate_uses_bayesian_live(kelly_config):
    rm = RiskManager(kelly_config)
    p = rm.effective_win_rate(
        "R_10",
        conviction=0.70,
        metrics={"live_n": 32, "live_wr": 0.35, "live_brier": 0.18, "live_ece": 0.05},
    )
    assert p < 0.70


def test_resolve_stake_regime_explore_vs_recover():
    assert resolve_stake_regime(pending_loss=0.0, consecutive_losses_linear=0) == "EXPLORE"
    assert resolve_stake_regime(pending_loss=1.0, consecutive_losses_linear=0) == "RECOVER"
    assert resolve_stake_regime(pending_loss=0.0, consecutive_losses_linear=1) == "RECOVER"


def test_recovery_infeasible_when_pending_exceeds_horizon():
    soft = {"amort_cycles_max": 5}
    assert is_recovery_infeasible(50.0, 4.20, 0.95, soft) is True
    assert is_recovery_infeasible(5.0, 4.20, 0.95, soft) is False


def test_soft_recovery_flags_infeasible_and_force_explore_unit():
    metrics: dict = {}
    soft = {
        "enabled": True,
        "max_safe_stake_cap": 4.20,
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "infeasible_force_explore": True,
    }
    stake = apply_soft_recovery_stake(
        pending_total=80.0,
        base_unit=1.0,
        consecutive_losses=2,
        previous_stake=2.0,
        bankroll=90.0,
        metrics=metrics,
        payout=0.95,
        soft_recovery=soft,
    )
    cap = max_safe_stake_cap(90.0, consecutive_losses_linear=2, soft_recovery=soft)
    assert metrics.get("recovery_infeasible") is True
    assert metrics.get("recovery_force_explore") is True
    assert stake <= 2.0 + 1e-9
    assert stake < cap


def test_soft_recovery_acc_below_floor_forces_explore():
    metrics = {"val_accuracy": 0.35}
    soft = {
        "enabled": True,
        "max_safe_stake_pct": 0.05,
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "infeasible_force_explore": True,
        "material_pending_min": 0.25,
    }
    stake = apply_soft_recovery_stake(
        pending_total=0.0,
        base_unit=10.0,
        consecutive_losses=4,
        previous_stake=20.0,
        bankroll=10000.0,
        metrics=metrics,
        payout=0.87,
        soft_recovery=soft,
    )
    assert metrics.get("recovery_acc_force_explore") is True
    assert metrics.get("recovery_force_explore") is True
    assert stake == pytest.approx(25.0)


def test_soft_recovery_acc_below_floor_waived_by_pending_uses_cover():
    metrics = {"val_accuracy": 0.35}
    soft = {
        "enabled": True,
        "max_safe_stake_pct": 0.05,
        "amort_cycles_min": 1,
        "amort_cycles_max": 1,
        "cover_multiple": 2.0,
        "infeasible_force_explore": True,
        "material_pending_min": 0.25,
    }
    stake = apply_soft_recovery_stake(
        pending_total=80.0,
        base_unit=10.0,
        consecutive_losses=4,
        previous_stake=20.0,
        bankroll=10000.0,
        metrics=metrics,
        payout=0.87,
        soft_recovery=soft,
    )
    assert metrics.get("recovery_force_explore") is False
    assert metrics.get("recovery_acc_force_explore") is False
    cover = 80.0 / 0.87 / 1.0 * 2.0
    assert stake + 1e-9 >= cover
    assert metrics.get("recovery_amort_cycles") == 1


def test_soft_recovery_live_wr_waived_by_pending_uses_cover():
    metrics = {"val_accuracy": 0.5424, "live_n": 3, "live_wr": 0.0}
    soft = {
        "enabled": True,
        "max_safe_stake_pct": 0.05,
        "max_safe_stake_pct_linear3": 0.03,
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "cover_multiple": 1.0,
        "infeasible_force_explore": True,
        "material_pending_min": 0.25,
        "live_evidence_force_explore_linear_min": 3,
        "live_evidence_force_explore_n_min": 2,
        "live_evidence_force_explore_wr_max": 0.58,
    }
    stake = apply_soft_recovery_stake(
        pending_total=85.0,
        base_unit=26.0,
        consecutive_losses=3,
        previous_stake=21.0,
        bankroll=10000.0,
        metrics=metrics,
        payout=0.87,
        soft_recovery=soft,
    )
    assert metrics.get("recovery_live_force_explore") is False
    assert metrics.get("recovery_force_explore") is False
    cover = 85.0 / 0.87 / 2.0
    assert stake + 1e-9 >= cover
    assert metrics.get("recovery_amort_cycles") == 2


def test_soft_recovery_adapted_waived_by_pending_uses_cover():
    metrics = {
        "val_accuracy": 0.5424,
        "live_n": 18,
        "live_wr": 0.56,
        "scale_adapted": True,
    }
    soft = {
        "enabled": True,
        "max_safe_stake_pct": 0.05,
        "max_safe_stake_pct_linear3": 0.03,
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "cover_multiple": 1.0,
        "infeasible_force_explore": True,
        "material_pending_min": 0.25,
        "adapted_force_explore": True,
        "adapted_force_explore_linear_min": 2,
        "live_evidence_force_explore_linear_min": 3,
        "live_evidence_force_explore_n_min": 2,
        "live_evidence_force_explore_wr_max": 0.58,
    }
    stake = apply_soft_recovery_stake(
        pending_total=88.0,
        base_unit=25.0,
        consecutive_losses=3,
        previous_stake=25.0,
        bankroll=10244.0,
        metrics=metrics,
        payout=0.87,
        soft_recovery=soft,
    )
    assert metrics.get("recovery_adapted_force_explore") is False
    assert metrics.get("recovery_force_explore") is False
    cover = 88.0 / 0.87 / 2.0
    assert stake + 1e-9 >= cover
    assert metrics.get("recovery_amort_cycles") == 2


def test_soft_recovery_cover_ge_cap_forces_explore():
    metrics: dict = {}
    soft = {
        "enabled": True,
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "max_safe_stake_pct": 0.025,
        "infeasible_force_explore": True,
    }
    stake = apply_soft_recovery_stake(
        pending_total=800.0,
        base_unit=10.0,
        consecutive_losses=2,
        previous_stake=20.0,
        bankroll=10000.0,
        metrics=metrics,
        payout=0.95,
        soft_recovery=soft,
    )
    cap = max_safe_stake_cap(10000.0, consecutive_losses_linear=2, soft_recovery=soft)
    assert metrics.get("recovery_infeasible") is True
    assert metrics.get("recovery_force_explore") is True
    assert float(metrics.get("recovery_cover_need", 0.0)) + 1e-12 >= cap
    assert stake == pytest.approx(25.0)
    assert stake < cap


def test_soft_recovery_infeasible_legacy_caps_without_force_explore():
    metrics: dict = {}
    soft = {
        "enabled": True,
        "max_safe_stake_cap": 4.20,
        "amort_cycles_min": 2,
        "amort_cycles_max": 5,
        "infeasible_force_explore": False,
    }
    stake = apply_soft_recovery_stake(
        pending_total=80.0,
        base_unit=1.0,
        consecutive_losses=2,
        previous_stake=2.0,
        bankroll=90.0,
        metrics=metrics,
        payout=0.95,
        soft_recovery=soft,
    )
    cap = max_safe_stake_cap(90.0, consecutive_losses_linear=2, soft_recovery=soft)
    assert metrics.get("recovery_infeasible") is True
    assert metrics.get("recovery_force_explore") is False
    assert stake <= cap + 1e-9
