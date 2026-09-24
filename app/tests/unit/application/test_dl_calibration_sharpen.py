"""Testes de sharpening termico nos logits contra colapso de variancia."""

from src.application.services.deep_learning.dl_calibration_sharpen import (
    apply_dynamic_temperature_sharpening,
    logit,
    resolve_sharpening_tau,
    sharpen_logit_temperature,
    sigmoid,
)


def test_logit_and_sigmoid_bounds():
    assert logit(0.5) == 0.0
    assert logit(1e-9) < -10.0
    assert logit(1.0 - 1e-9) > 10.0
    assert sigmoid(0.0) == 0.5
    assert sigmoid(5.0) > 0.99
    assert sigmoid(-5.0) < 0.01


def test_sharpen_logit_temperature_expansion():
    p = 0.515
    p_sharp = sharpen_logit_temperature(p, 0.40)
    assert p_sharp > p
    p_put = 0.485
    p_put_sharp = sharpen_logit_temperature(p_put, 0.40)
    assert p_put_sharp < p_put
    assert sharpen_logit_temperature(0.5, 0.30) == 0.5


def test_resolve_sharpening_tau_volatility():
    tau_norm = resolve_sharpening_tau(0.40, vol_ratio=1.0)
    assert abs(tau_norm - 0.40) < 1e-6
    tau_high_vol = resolve_sharpening_tau(0.40, vol_ratio=4.0)
    assert tau_high_vol < tau_norm
    tau_low_vol = resolve_sharpening_tau(0.40, vol_ratio=0.1)
    assert tau_low_vol <= 0.40


def test_apply_dynamic_temperature_sharpening():
    p_spread = 0.65
    res, applied = apply_dynamic_temperature_sharpening(p_spread, margin_threshold=0.035)
    assert not applied
    assert res == p_spread

    p_collapsed = 0.51
    res_collapsed, applied_collapsed = apply_dynamic_temperature_sharpening(p_collapsed, margin_threshold=0.035)
    assert applied_collapsed
    assert res_collapsed > p_collapsed

    res_forced, applied_forced = apply_dynamic_temperature_sharpening(p_spread, margin_threshold=0.035, force=True)
    assert applied_forced
    assert res_forced > p_spread


def test_apply_calibrator_stable_with_sharpen():
    from src.application.services.deep_learning.dl_calibration import (
        CalibratorState,
        apply_calibrator_stable,
    )

    calibrator = CalibratorState(method="platt", platt_a=1.0, platt_b=0.0)
    p_sharp = apply_calibrator_stable(0.52, calibrator, sharpen=True, sharpening_tau=0.40)
    assert p_sharp > 0.52


def test_build_prediction_entry_with_sharpening():
    from types import SimpleNamespace

    import numpy as np

    from src.application.services.deep_learning.dl_predict_build import build_prediction_entry

    orch = SimpleNamespace(config={})
    prices = np.full(50, 100.0)
    series = {"bb_width": [0.05], "vol_ratio_short_long": [1.0]}
    runtime = {"val_accuracy": 0.55, "deploy_ok": True}
    params = {
        "calibration": {"sharpening_enabled": True, "sharpening_tau": 0.40, "max_calibrated_raw_gap": 0.20},
        "indicators": {
            "windows": {"bb_window": 20},
            "multipliers": {"bb_std_mult": 2.0},
            "congestion": {"min_bars": 20, "bb_width_max": 0.04, "adx_max": 0.15},
        },
    }
    dynamic = SimpleNamespace(call_threshold=0.55, put_threshold=0.45, min_edge=0.04, regime_score=1.0)
    dynamic_cfg = {}
    entry = build_prediction_entry(
        orch,
        "1HZ75V",
        prices,
        series,
        runtime,
        params,
        train_loss=0.1,
        direction=None,
        prob=0.537,
        raw_prob=0.537,
        dynamic=dynamic,
        dynamic_cfg=dynamic_cfg,
        call_threshold=0.55,
        put_threshold=0.45,
        exec_cfg={},
        val_accuracy=0.55,
    )
    cal_p = entry["metrics"]["calibrated_prob"]
    assert cal_p > 0.58
