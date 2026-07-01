import pytest

from src.domain.risk.martingale_sizing import (
    _effective_step_frac,
    assert_martingale_safety_cap,
    calculate_vol_adjusted_martingale,
    martingale_defer_active,
    martingale_progressive_slice_cycles,
    martingale_stake,
    resolve_mode_stake,
)


def _kelly_cfg(*, enabled=True):
    return {
        "martingale_vol_adjust_enabled": enabled,
        "martingale_vol_defer_ratio": 1.10,
        "martingale_deferred_max_recovery_bankroll_pct": 0.025,
        "martingale_max_recovery_bankroll_pct": 0.05,
        "martingale_vol_losses_min": 2,
        "martingale_target_fraction": 1.0,
        "martingale_recovery_step_fraction": 1.0,
        "martingale_max_stake_multiplier": 0.0,
        "min_stake_pct": 0.0,
        "stop_win_kelly_enabled": False,
    }


def test_vol_adjust_skipped_when_losses_below_min():
    cfg = _kelly_cfg()
    base = 500.0
    assert calculate_vol_adjusted_martingale(
        base,
        bankroll=10000.0,
        vol_ratio=1.23,
        consecutive_losses=1,
        kelly_config=cfg,
    ) == pytest.approx(base)


def test_vol_adjust_sqrt_and_bankroll_cap():
    cfg = _kelly_cfg()
    bankroll = 10000.0
    linear = martingale_stake(
        bankroll,
        400.0,
        50.0,
        0.95,
        cfg,
        1.0,
        vol_ratio=1.0,
        consecutive_losses=0,
    )
    adjusted = calculate_vol_adjusted_martingale(
        linear,
        bankroll=bankroll,
        vol_ratio=1.23,
        consecutive_losses=3,
        kelly_config=cfg,
    )
    assert adjusted < linear
    assert adjusted <= bankroll * 0.05 + 1e-9


def test_deferred_step_frac_when_vol_high():
    cfg = _kelly_cfg()
    full = martingale_stake(
        10000.0,
        200.0,
        30.0,
        0.95,
        cfg,
        1.0,
        vol_ratio=1.0,
        consecutive_losses=0,
    )
    deferred = martingale_stake(
        10000.0,
        200.0,
        30.0,
        0.95,
        cfg,
        1.0,
        vol_ratio=1.20,
        consecutive_losses=3,
    )
    assert deferred < full
    assert _effective_step_frac(1.0, vol_ratio=1.20, consecutive_losses=3, kelly_config=cfg) == 0.5


def test_effective_step_frac_skipped_when_vol_adjust_disabled():
    cfg = _kelly_cfg(enabled=False)
    assert _effective_step_frac(0.85, vol_ratio=1.20, consecutive_losses=3, kelly_config=cfg) == 0.85


def test_vol_adjust_disabled_preserves_legacy():
    cfg = _kelly_cfg(enabled=False)
    base = 300.0
    assert calculate_vol_adjusted_martingale(
        base,
        bankroll=5000.0,
        vol_ratio=1.5,
        consecutive_losses=5,
        kelly_config=cfg,
    ) == pytest.approx(base)


def test_resolve_mode_stake_applies_vol_adjust():
    cfg = _kelly_cfg()
    bankroll = 8000.0
    stake, _, mode = resolve_mode_stake(
        martingale_active=True,
        bankroll=bankroll,
        loss_to_recover=350.0,
        kelly_base=40.0,
        payout=0.95,
        kelly_config=cfg,
        stake_min=1.0,
        vol_ratio=1.23,
        consecutive_losses=3,
    )
    assert mode == "MARTINGALE"
    assert stake <= bankroll * 0.05 + 1e-9


def test_defer_active_at_vol_115_with_two_losses():
    cfg = _kelly_cfg()
    assert martingale_defer_active(1.15, 2, cfg) is True
    assert _effective_step_frac(1.0, vol_ratio=1.15, consecutive_losses=2, kelly_config=cfg) == 0.5


def test_defer_inactive_below_vol_threshold():
    cfg = _kelly_cfg()
    assert martingale_defer_active(1.08, 3, cfg) is False


def test_deferred_recovery_capped_at_two_point_five_pct_bankroll():
    cfg = _kelly_cfg()
    bankroll = 10000.0
    base = 800.0
    adjusted = calculate_vol_adjusted_martingale(
        base,
        bankroll=bankroll,
        vol_ratio=1.20,
        consecutive_losses=3,
        kelly_config=cfg,
    )
    assert adjusted <= bankroll * 0.025 + 1e-9


def test_resolve_mode_stake_deferred_cap_at_two_point_five_pct():
    cfg = _kelly_cfg()
    bankroll = 10000.0
    stake, _, mode = resolve_mode_stake(
        martingale_active=True,
        bankroll=bankroll,
        loss_to_recover=500.0,
        kelly_base=40.0,
        payout=0.95,
        kelly_config=cfg,
        stake_min=1.0,
        vol_ratio=1.15,
        consecutive_losses=2,
    )
    assert mode == "MARTINGALE"
    assert stake <= bankroll * 0.025 + 1e-9


def test_progressive_slice_cycles_at_three_and_four_losses():
    cfg = _kelly_cfg()
    cfg["martingale_safety_losses_min"] = 3
    assert martingale_progressive_slice_cycles(2, cfg) == 1
    assert martingale_progressive_slice_cycles(3, cfg) == 2
    assert martingale_progressive_slice_cycles(4, cfg) == 3
    assert martingale_progressive_slice_cycles(6, cfg) == 3


def test_assert_martingale_safety_cap_four_pct():
    cfg = _kelly_cfg()
    cfg["martingale_hard_cap_bankroll_pct"] = 0.04
    bankroll = 20000.0
    capped = assert_martingale_safety_cap(900.0, bankroll=bankroll, kelly_config=cfg)
    assert capped == pytest.approx(800.0)
    assert assert_martingale_safety_cap(500.0, bankroll=bankroll, kelly_config=cfg) == pytest.approx(500.0)


def test_assert_safety_cap_zero_bankroll_and_disabled_cap():
    cfg = _kelly_cfg()
    assert assert_martingale_safety_cap(10.0, bankroll=0.0, kelly_config=cfg) == pytest.approx(10.0)
    cfg["martingale_hard_cap_bankroll_pct"] = 0.0
    assert assert_martingale_safety_cap(75.0, bankroll=5000.0, kelly_config=cfg) == pytest.approx(75.0)


def test_c0012_scenario_slice_and_cap_prevent_full_recovery_stake():
    cfg = _kelly_cfg()
    cfg["martingale_safety_losses_min"] = 3
    cfg["martingale_hard_cap_bankroll_pct"] = 0.04
    cfg["martingale_recovery_step_fraction"] = 1.0
    cfg["martingale_vol_adjust_enabled"] = False
    bankroll = 20764.0
    pending = 830.57
    linear_stake, _, _ = resolve_mode_stake(
        martingale_active=True,
        bankroll=bankroll,
        loss_to_recover=pending,
        kelly_base=50.0,
        payout=0.95,
        kelly_config=cfg,
        stake_min=1.0,
        vol_ratio=1.0,
        consecutive_losses=0,
    )
    safe_stake, _, mode = resolve_mode_stake(
        martingale_active=True,
        bankroll=bankroll,
        loss_to_recover=pending,
        kelly_base=50.0,
        payout=0.95,
        kelly_config=cfg,
        stake_min=1.0,
        vol_ratio=1.0,
        consecutive_losses=3,
    )
    assert mode == "MARTINGALE"
    assert safe_stake <= bankroll * 0.04 + 1e-6
    assert safe_stake < linear_stake
