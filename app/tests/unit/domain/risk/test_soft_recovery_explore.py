"""Cobertura do EXPLORE forçado sem U sticky nem cover."""

from __future__ import annotations

import pytest

from src.domain.risk.soft_recovery_explore import (
    damped_cover_stake,
    force_early_explore_reason,
    forced_explore_stake,
    mark_forced_explore_metrics,
    soft_early_infeasible,
    soft_floor_scale,
)


def test_soft_floor_scale_always_one():
    assert soft_floor_scale(None) == pytest.approx(1.0)
    assert soft_floor_scale({"loss_clf_soft": True}) == pytest.approx(1.0)


def test_damped_cover_applies_target_proximity_when_amort_gt_one():
    soft = {"amort_cycles_min": 2, "amort_cycles_max": 4, "cover_multiple": 1.5}
    stake, cover, amort = damped_cover_stake(
        pending=90.0,
        consecutive_losses=1,
        payout=0.95,
        risk_params=None,
        soft=soft,
        target=100.0,
        pnl=50.0,
    )
    assert amort == 3
    assert cover == pytest.approx(90.0 / 0.95 / 3.0 * 1.5)
    assert stake < cover


def test_forced_explore_infeasible_ignores_pending_cover():
    metrics: dict = {}
    soft = {"amort_cycles_min": 2, "amort_cycles_max": 4, "cover_multiple": 1.5}
    stake = forced_explore_stake(
        bankroll=10000.0,
        pending=800.0,
        material_pending=True,
        consecutive_losses=2,
        payout=0.95,
        risk_params=None,
        soft=soft,
        target=0.0,
        pnl=0.0,
        cap=250.0,
        metrics=metrics,
        reason="infeasible",
    )
    assert stake == pytest.approx(25.0)
    assert metrics["recovery_explore_used_cover"] is False
    assert metrics["recovery_force_explore_reason"] == "infeasible"


def test_forced_explore_material_pending_uses_floor_not_cover():
    metrics: dict = {}
    soft = {"amort_cycles_min": 2, "amort_cycles_max": 4, "cover_multiple": 1.5}
    stake = forced_explore_stake(
        bankroll=11000.0,
        pending=90.0,
        material_pending=True,
        consecutive_losses=1,
        payout=0.95,
        risk_params=None,
        soft=soft,
        target=0.0,
        pnl=0.0,
        cap=550.0,
        metrics=metrics,
        reason="neg_edge",
    )
    cover = 90.0 / 0.95 / 3.0 * 1.5
    floor = 11000.0 * 0.0025
    assert stake == pytest.approx(floor)
    assert stake < cover
    assert metrics["recovery_explore_used_cover"] is False
    assert metrics["recovery_force_explore_reason"] == "neg_edge"


def test_force_early_explore_reason_priority():
    assert (
        force_early_explore_reason(
            near_stop_win=True,
            low_hurst_noise=True,
            chop_neg_dampen=True,
            quality_force_explore=True,
        )
        == "near_stop"
    )
    assert (
        force_early_explore_reason(
            near_stop_win=False,
            low_hurst_noise=False,
            chop_neg_dampen=False,
            quality_force_explore=True,
        )
        == "quality"
    )


def test_soft_early_infeasible_false_without_pending():
    soft = {"amort_cycles_min": 2, "amort_cycles_max": 4, "cover_multiple": 1.5}
    assert (
        soft_early_infeasible(
            pending=0.0,
            material_pending=False,
            consecutive_losses=2,
            payout=0.95,
            risk_params=None,
            soft=soft,
            soft_recovery=soft,
            cap=250.0,
        )
        is False
    )


def test_mark_forced_explore_metrics_noop_when_metrics_none():
    mark_forced_explore_metrics(
        None,
        consecutive_losses=1,
        previous_stake=10.0,
        unit=5.0,
        material_pending=True,
        near_stop_win=False,
        low_hurst_noise=False,
        chop_neg_dampen=True,
        acc_force_explore=False,
        live_force_explore=False,
        adapted_force_explore=False,
        quality_force_explore=False,
        soft_infeasible=True,
    )
