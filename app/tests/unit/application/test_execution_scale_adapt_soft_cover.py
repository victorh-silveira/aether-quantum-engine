"""Testes de soft cover SCALE e waiver de EXPLORE sob pending."""

import pytest

from src.domain.models.trade import TradeDirection
from src.domain.risk.dlambert_sizing import resolve_dlambert_stake
from src.domain.risk.risk_stake_calc_helpers import apply_loss_clf_soft_stake_cap, apply_scale_stake_cap


def test_stake_cap_discord_and_noop():
    assert apply_scale_stake_cap(100.0, 10000.0, None) == 100.0
    assert apply_scale_stake_cap(100.0, 10000.0, {"scale_discordance": False}) == 100.0
    assert apply_scale_stake_cap(100.0, 10000.0, {"scale_adapted": True, "scale_max_stake_pct": None}) == 100.0
    capped = apply_scale_stake_cap(100.0, 10000.0, {"scale_adapted": True, "scale_max_stake_pct": 0.005})
    assert capped == 50.0
    assert apply_scale_stake_cap(10.0, 10000.0, {"scale_adapted": True, "scale_max_stake_pct": "bad"}) == 10.0
    waived = apply_scale_stake_cap(
        100.0,
        10000.0,
        {"scale_adapted": True, "scale_max_stake_pct": 0.005},
        pending_total=51.0,
        soft_recovery={"pending_waives_scale_explore": True, "material_pending_min": 0.5},
    )
    assert waived == 100.0


def test_loss_clf_soft_stake_cap_explore_only_waived_by_pending():
    metrics = {"loss_clf_soft": True, "loss_clf_soft_max_stake_pct": 0.0025}
    soft = {"pending_waives_scale_explore": True, "material_pending_min": 0.5}
    assert apply_loss_clf_soft_stake_cap(94.0, 9773.0, metrics) == pytest.approx(24.4325)
    assert apply_loss_clf_soft_stake_cap(20.0, 9773.0, metrics) == 20.0
    assert apply_loss_clf_soft_stake_cap(94.0, 9773.0, {"loss_clf_soft": False}) == 94.0
    assert apply_loss_clf_soft_stake_cap(94.0, 9773.0, None) == 94.0
    assert apply_loss_clf_soft_stake_cap(94.0, 9773.0, {"loss_clf_soft": True}) == 94.0
    waived = apply_loss_clf_soft_stake_cap(
        94.0,
        9773.0,
        metrics,
        pending_total=51.0,
        soft_recovery=soft,
    )
    assert waived == pytest.approx(94.0)


def test_resolve_dlambert_pending_waives_scale_adapted_uses_soft_cover():
    class RM:
        soft_recovery_config = {
            "enabled": True,
            "cover_enabled": True,
            "max_safe_stake_cap": 300.0,
            "max_safe_stake_pct": 0.05,
            "pending_waives_scale_explore": True,
            "material_pending_min": 0.5,
            "infeasible_force_explore": True,
            "amort_cycles_min": 2,
            "amort_cycles_max": 5,
        }
        last_loss_stake = 50.0
        dlambert_unit = 20.0
        risk_params = {"payout_estimate": 0.82}
        total_session_profit = 0.0
        daily_stop_win_target = 0.0

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=10000.0,
        kelly_base=20.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=1,
        pending_total=51.0,
        payout=0.82,
        dl_metrics={"scale_adapted": True, "execute": True},
        f_star=0.01,
    )
    assert tag == "D'ALEMBERT"
    cover = 51.0 / 0.82 / 4.0 * 1.1
    assert stake == pytest.approx(cover, rel=1e-2)


def test_resolve_dlambert_skips_dal_on_scale_adapted_without_pending():
    class RM:
        soft_recovery_config = {
            "enabled": True,
            "max_safe_stake_cap": 300.0,
            "max_safe_stake_pct": 0.03,
            "pending_waives_scale_explore": True,
            "material_pending_min": 0.5,
        }
        last_loss_stake = 50.0
        dlambert_unit = 1.0
        risk_params = {}
        total_session_profit = 0.0
        daily_stop_win_target = 0.0

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=5000.0,
        kelly_base=20.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=3,
        pending_total=0.0,
        payout=0.87,
        dl_metrics={"scale_adapted": True, "execute": True},
        f_star=0.01,
    )
    assert tag == "KELLY"
    assert stake > 0.0


def test_resolve_dlambert_skips_dal_on_scale_force_explore_when_waiver_off():
    class RM:
        soft_recovery_config = {
            "enabled": True,
            "max_safe_stake_cap": 300.0,
            "max_safe_stake_pct": 0.03,
            "pending_waives_scale_explore": False,
            "material_pending_min": 0.5,
        }
        last_loss_stake = 50.0
        dlambert_unit = 1.0
        risk_params = {}
        total_session_profit = 0.0
        daily_stop_win_target = 0.0

    stake, tag = resolve_dlambert_stake(
        recovery_active=True,
        bankroll=5000.0,
        kelly_base=20.0,
        dlambert_config={"dlambert_enabled": True},
        rm=RM(),
        consecutive_losses_linear=3,
        pending_total=80.0,
        payout=0.87,
        dl_metrics={"scale_force_explore": True, "execute": True},
        f_star=0.01,
    )
    assert tag == "KELLY"
    assert stake > 0.0


def test_finalize_adapts_direction_under_raw_extreme():
    from unittest.mock import patch

    import numpy as np

    from src.application.services.execution_direction_resolver import _finalize_execution_metrics

    entry = {"metrics": {}}
    metrics = {
        "calibration_mode": "raw_extreme",
        "flow_features": {"price_velocity": 1.0},
        "kelly_fraction_scale": 1.0,
        "calibrated_prob": 0.65,
        "trade_score": 0.52,
        "conviction": 0.52,
    }

    class Stream:
        def get_numpy_series(self, _symbol, _field="close"):
            return np.array([1.0, 1.05, 1.1, 1.15, 1.2])

        def get_mini_numpy_series(self, _symbol, field="close"):
            if field == "open":
                return np.array([1.0, 1.0, 1.0, 1.0, 1.0])
            return np.array([1.0, 1.05, 1.1, 1.15, 1.2])

        def get_micro_numpy_series(self, _symbol, field="close"):
            if field == "open":
                return np.array([1.0, 1.0, 1.0])
            return np.array([0.95, 1.05, 1.15])

        tick_buffer = None

    orch = type(
        "O",
        (),
        {
            "stream": Stream(),
            "config": {"orchestrator": {"execution": {"scale_vision": {"fusion_enabled": False}}}},
        },
    )()
    with (
        patch(
            "src.application.services.execution_direction_resolver.apply_signal_skip_gates",
            return_value=False,
        ),
        patch(
            "src.application.services.execution_direction_resolver.apply_loss_classifier_gate",
            return_value=False,
        ),
    ):
        direction, out = _finalize_execution_metrics(
            entry,
            metrics,
            TradeDirection.PUT,
            0.2,
            0.01,
            meta_applied=False,
            score=0.55,
            symbol="R_10",
            orch=orch,
        )
    assert direction == TradeDirection.CALL
    assert out["scale_adapted"] is True
    assert out["tcn_direction"] == "PUT"
    assert out["execution_candidate_ready"] is True
    assert out["scale_force_explore"] is True
    assert out["scale_tape_strong"] is True
    assert out.get("scale_kelly_side_synced") is True
    assert float(out.get("conviction") or 0.0) >= 0.55
