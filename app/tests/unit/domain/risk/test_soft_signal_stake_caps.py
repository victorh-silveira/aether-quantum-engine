"""Caps de stake sob soft de sinal e discord SCALE."""

import pytest

from src.application.services.loss_classifier_flip import apply_soft_kelly
from src.domain.risk.risk_stake_calc_helpers import (
    apply_loss_clf_soft_stake_cap,
    apply_post_kelly_stake_caps,
    apply_signal_soft_stake_cap,
)


def test_cal_margin_soft_without_explicit_pct_does_not_crush_stake():
    metrics = {"cal_margin_soft": True}
    capped = apply_signal_soft_stake_cap(200.0, 10000.0, metrics, pending_total=0.0)
    assert capped == pytest.approx(200.0)


def test_soft_signal_cap_honors_explicit_pct():
    metrics = {"cal_margin_soft": True, "soft_signal_max_stake_pct": 0.01}
    capped = apply_signal_soft_stake_cap(500.0, 10000.0, metrics, pending_total=0.0)
    assert capped == pytest.approx(100.0)


def test_loss_clf_soft_cap_uses_configured_pct():
    metrics = {"loss_clf_soft": True, "loss_clf_soft_max_stake_pct": 0.05}
    capped = apply_loss_clf_soft_stake_cap(500.0, 10000.0, metrics, pending_total=0.0)
    assert capped == pytest.approx(500.0)


def test_flip_block_skips_soft_max_stake_pct():
    metrics = {"loss_clf_flip_blocked": "seed_candle", "kelly_fraction_scale": 1.0}
    apply_soft_kelly(
        metrics,
        0.40,
        p_loss=0.95,
        cfg={"soft_max_stake_pct_high": 0.05},
    )
    assert metrics["loss_clf_soft"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.40)
    assert "loss_clf_soft_max_stake_pct" not in metrics


def test_soft_signal_cap_waived_with_material_pending():
    metrics = {"cal_margin_soft": True, "soft_signal_max_stake_pct": 0.01}
    soft = {"material_pending_min": 0.25, "pending_waives_scale_explore": True}
    capped = apply_signal_soft_stake_cap(
        200.0,
        10000.0,
        metrics,
        pending_total=100.0,
        soft_recovery=soft,
    )
    assert capped == pytest.approx(200.0)


def test_soft_signal_cap_rejects_invalid_or_nonpositive_pct():
    metrics = {"cal_margin_soft": True, "soft_signal_max_stake_pct": "x"}
    assert apply_signal_soft_stake_cap(200.0, 10000.0, metrics) == pytest.approx(200.0)
    metrics["soft_signal_max_stake_pct"] = 0.0
    assert apply_signal_soft_stake_cap(200.0, 10000.0, metrics) == pytest.approx(200.0)


def test_post_kelly_applies_discord_cap():
    metrics = {
        "scale_discordance": True,
        "scale_max_stake_pct": 0.01,
    }
    capped = apply_post_kelly_stake_caps(500.0, 10000.0, metrics, pending_total=0.0)
    assert capped == pytest.approx(100.0)


def test_post_kelly_discord_cap_allows_single_strike_five_pct():
    metrics = {
        "scale_discordance": True,
        "scale_max_stake_pct": 0.05,
    }
    bankroll = 9665.64
    single_strike = bankroll * 0.0507
    capped = apply_post_kelly_stake_caps(single_strike, bankroll, metrics, pending_total=0.0)
    assert capped == pytest.approx(bankroll * 0.05)
    assert capped / bankroll == pytest.approx(0.05)
