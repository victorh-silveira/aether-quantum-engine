"""Testes unitarios para funcoes matematicas de logit sharpening."""

from src.domain.math.logit_sharpen import (
    apply_dynamic_temperature_sharpening,
    logit,
    resolve_sharpening_tau,
    sharpen_logit_temperature,
    sigmoid,
)


def test_logit_sigmoid_roundtrip():
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert abs(sigmoid(logit(p)) - p) < 1e-6
    assert logit(1e-9) < -15.0
    assert logit(1.0 - 1e-9) > 15.0
    assert sigmoid(0.0) == 0.5
    assert sigmoid(10.0) > 0.999
    assert sigmoid(-10.0) < 0.001


def test_sharpen_logit_temperature_effects():
    p = 0.515
    sharp = sharpen_logit_temperature(p, 0.40)
    assert sharp > p
    p_put = 0.485
    sharp_put = sharpen_logit_temperature(p_put, 0.40)
    assert sharp_put < p_put
    assert sharpen_logit_temperature(0.5, 0.30) == 0.5


def test_resolve_sharpening_tau_bounds():
    tau = resolve_sharpening_tau(0.40, vol_ratio=1.0)
    assert abs(tau - 0.40) < 1e-6
    tau_high = resolve_sharpening_tau(0.40, vol_ratio=4.0)
    assert tau_high < 0.40
    tau_low = resolve_sharpening_tau(0.40, vol_ratio=0.01)
    assert 0.15 <= tau_low <= 0.40


def test_apply_dynamic_temperature_sharpening_modes():
    p_wide = 0.65
    res, applied = apply_dynamic_temperature_sharpening(p_wide, margin_threshold=0.035)
    assert not applied
    assert res == p_wide

    p_tight = 0.51
    res_tight, applied_tight = apply_dynamic_temperature_sharpening(p_tight, margin_threshold=0.035)
    assert applied_tight
    assert res_tight > p_tight

    res_forced, applied_forced = apply_dynamic_temperature_sharpening(p_wide, margin_threshold=0.035, force=True)
    assert applied_forced
    assert res_forced > p_wide
