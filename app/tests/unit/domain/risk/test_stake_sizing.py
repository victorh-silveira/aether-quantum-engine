import pytest

from src.domain.risk.stake_sizing import (
    _resolve_stop_win_max_stake_pct,
    apply_symbol_stake_cap,
    compute_single_strike_kelly_base,
    conviction_stop_win_weight,
    enrich_metrics_conviction,
    resolve_cycle_stake_scale,
    resolve_stake_conviction,
)


def test_resolve_stake_conviction_from_raw_when_score_zero():
    metrics = {"trade_score": 0.0, "raw_prob": 0.51}
    assert resolve_stake_conviction(metrics) == pytest.approx(0.51, abs=1e-6)
    assert resolve_stake_conviction({"trade_score": 0.55, "raw_prob": 0.48}) == pytest.approx(0.55, abs=1e-6)


def test_enrich_metrics_conviction_fills_zero_score():
    metrics = {"trade_score": 0.0, "raw_prob": 0.52}
    enrich_metrics_conviction(metrics)
    assert metrics["trade_score"] == pytest.approx(0.52, abs=1e-6)
    assert metrics["conviction"] == pytest.approx(0.52, abs=1e-6)


def test_resolve_stake_conviction_from_raw_conviction():
    metrics = {"trade_score": 0.0, "raw_conviction": 0.52}
    assert resolve_stake_conviction(metrics) == pytest.approx(0.52, abs=1e-6)


def test_enrich_metrics_conviction_uses_raw_conviction():
    metrics = {"trade_score": 0.0, "raw_conviction": 0.54}
    enrich_metrics_conviction(metrics)
    assert metrics["trade_score"] == pytest.approx(0.54, abs=1e-6)


def test_compute_single_strike_returns_kelly_when_conviction_low():
    kelly = compute_single_strike_kelly_base(
        50.0,
        1000.0,
        0.95,
        0.40,
        {"large_account_stop_win_pct": 4.0},
        {},
        1000.0,
        0.0,
        has_active_contracts=False,
    )
    assert kelly == 50.0


def test_compute_single_strike_keeps_kelly_when_boost_not_greater():
    kelly = compute_single_strike_kelly_base(
        5000.0,
        10000.0,
        0.95,
        0.8,
        {"large_account_stop_win_pct": 10.0, "small_account_threshold": 0.0},
        {"max_stake_pct": 0.01},
        10000.0,
        0.0,
        has_active_contracts=False,
    )
    assert kelly == 5000.0


def test_compute_single_strike_targets_stop_win_pct():
    risk = {"large_account_stop_win_pct": 4.0, "small_account_threshold": 50.0}
    cfg = {
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_kelly_conviction_strong": 0.72,
        "stop_win_kelly_min_fraction": 0.42,
        "stop_win_kelly_max_fraction": 1.0,
    }
    weak = compute_single_strike_kelly_base(
        1.16,
        1168.0,
        0.95,
        0.46,
        risk,
        cfg,
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    weight = conviction_stop_win_weight(0.46, cfg)
    assert weak == pytest.approx((46.72 / 0.95) * weight, abs=0.5)
    strong = compute_single_strike_kelly_base(
        1.16,
        1168.0,
        0.95,
        0.75,
        risk,
        cfg,
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    assert strong == pytest.approx(46.72 / 0.95, abs=0.5)


def test_resolve_cycle_stake_scale_m5():
    scale = resolve_cycle_stake_scale(
        {"cycle_stake_baseline_seconds": 60, "cycle_stake_exponent": 0.55},
        {"orchestrator": {"cycle_interval_seconds": 300}},
    )
    assert scale == pytest.approx((300 / 60) ** 0.55, rel=1e-6)
    contract_scale = resolve_cycle_stake_scale(
        {
            "cycle_stake_baseline_seconds": 60,
            "cycle_stake_exponent": 0.55,
            "cycle_stake_use_contract_duration": True,
        },
        {"params": {"duration": 300, "duration_unit": "s"}},
    )
    assert contract_scale == pytest.approx((300 / 60) ** 0.55, rel=1e-6)
    assert resolve_cycle_stake_scale(
        {
            "cycle_stake_baseline_seconds": 60,
            "cycle_stake_use_contract_duration": True,
        },
        {"params": {"duration": 5, "duration_unit": "m"}},
    ) == pytest.approx((300 / 60) ** 0.55, rel=1e-6)
    assert resolve_cycle_stake_scale(
        {"cycle_stake_baseline_seconds": 60, "cycle_stake_use_contract_duration": True},
        {"params": {"duration": 40, "duration_unit": "t"}},
    ) == pytest.approx((80 / 60) ** 0.55, rel=1e-6)
    assert resolve_cycle_stake_scale(
        {"cycle_stake_baseline_seconds": 60, "cycle_stake_use_contract_duration": True},
        {"params": {"duration": 1, "duration_unit": "d"}},
    ) == pytest.approx((86400 / 60) ** 0.55, rel=1e-6)
    assert resolve_cycle_stake_scale(
        {"cycle_stake_baseline_seconds": 60, "cycle_stake_use_contract_duration": True},
        {"params": {"duration": 15, "duration_unit": "x"}},
    ) == pytest.approx((900 / 60) ** 0.55, rel=1e-6)
    assert resolve_cycle_stake_scale({}, {"orchestrator": {"cycle_interval_seconds": 60}}) == 1.0
    assert (
        resolve_cycle_stake_scale(
            {"cycle_stake_scale_enabled": False}, {"orchestrator": {"cycle_interval_seconds": 300}}
        )
        == 1.0
    )
    assert (
        resolve_cycle_stake_scale(
            {"cycle_stake_baseline_seconds": 0}, {"orchestrator": {"cycle_interval_seconds": 300}}
        )
        == 1.0
    )


def test_compute_single_strike_cycles_target_reduces_stake():
    risk = {"large_account_stop_win_pct": 4.0, "small_account_threshold": 50.0}
    base_cfg = {
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_kelly_conviction_strong": 0.75,
        "stop_win_kelly_min_fraction": 0.12,
        "stop_win_kelly_max_fraction": 0.38,
        "stop_win_kelly_cycles_target": 1.0,
    }
    full = compute_single_strike_kelly_base(
        1.16,
        1168.0,
        0.95,
        0.50,
        risk,
        base_cfg,
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    damped_cfg = {**base_cfg, "stop_win_kelly_cycles_target": 2.75}
    damped = compute_single_strike_kelly_base(
        1.16,
        1168.0,
        0.95,
        0.50,
        risk,
        {**damped_cfg, "cycle_stake_scale_enabled": False},
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    assert damped < full
    assert damped == pytest.approx(full / 2.75, abs=0.5)


def test_apply_symbol_stake_cap_limits_r10():
    cfg = {"symbol_max_stake_pct": {"R_10": 0.009}}
    capped = apply_symbol_stake_cap(126.0, 10545.0, "R_10", cfg)
    assert capped == pytest.approx(10545.0 * 0.009, abs=0.02)
    assert apply_symbol_stake_cap(80.0, 10545.0, "R_50", cfg) == 80.0


def test_compute_single_strike_scales_with_m5_cycle():
    risk = {
        "large_account_stop_win_pct": 4.0,
        "small_account_threshold": 50.0,
        "orchestrator": {"cycle_interval_seconds": 300},
    }
    cfg = {
        "stop_win_kelly_enabled": True,
        "stop_win_kelly_min_conviction": 0.45,
        "stop_win_kelly_conviction_strong": 0.75,
        "stop_win_kelly_min_fraction": 0.12,
        "stop_win_kelly_max_fraction": 0.38,
        "stop_win_kelly_cycles_target": 1.0,
        "cycle_stake_baseline_seconds": 60,
        "cycle_stake_exponent": 0.55,
    }
    base = compute_single_strike_kelly_base(
        1.16,
        1168.0,
        0.95,
        0.50,
        risk,
        {**cfg, "cycle_stake_scale_enabled": False},
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    scaled = compute_single_strike_kelly_base(
        1.16,
        1168.0,
        0.95,
        0.50,
        risk,
        cfg,
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    assert scaled > base
    assert scaled == pytest.approx(base * resolve_cycle_stake_scale(cfg, risk), abs=0.5)


def test_compute_single_strike_disabled_when_flag_off():
    kelly = compute_single_strike_kelly_base(
        12.0,
        1168.0,
        0.95,
        0.60,
        {"large_account_stop_win_pct": 4.0},
        {"stop_win_kelly_enabled": False},
        1168.0,
        0.0,
        has_active_contracts=False,
    )
    assert kelly == 12.0


def test_resolve_stop_win_max_stake_pct_default_one_percent():
    pct = _resolve_stop_win_max_stake_pct({}, {}, 0.95)
    assert pct == pytest.approx(0.01 / 0.95, rel=1e-6)


def test_resolve_stop_win_max_stake_pct_from_stop_win():
    pct = _resolve_stop_win_max_stake_pct({"large_account_stop_win_pct": 4.0}, {}, 0.95)
    assert pct == pytest.approx(0.04 / 0.95, rel=1e-6)


def test_resolve_stop_win_max_stake_pct_explicit_override():
    pct = _resolve_stop_win_max_stake_pct({}, {"stop_win_max_stake_pct": 0.03}, 0.95)
    assert pct == 0.03


def test_resolve_stop_win_max_stake_pct_without_payout():
    pct = _resolve_stop_win_max_stake_pct({"large_account_stop_win_pct": 4.0}, {}, 0.0)
    assert pct == pytest.approx(0.04, rel=1e-6)
