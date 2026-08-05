from src.application.services.deep_learning.dl_calibration import (
    CalibratorState,
    apply_calibrator,
    calibrator_from_dict,
    calibrator_to_dict,
)
from src.application.services.deep_learning.dl_calibration_fit import (
    _select_best_calibrator,
    fit_calibrator,
)
from src.application.services.deep_learning.dl_calibration_isotonic import apply_isotonic, fit_isotonic
from src.application.services.execution_volatility_threshold import (
    _median_tail,
    _vol_component,
    resolve_dynamic_threshold_bundle,
    resolve_dynamic_thresholds,
    volatility_regime_score,
)


def test_fit_isotonic_empty():
    assert fit_isotonic([], []) == ((), ())


def test_fit_isotonic_pools_violations():
    xs, ys = fit_isotonic([0.1, 0.2, 0.3, 0.4, 0.5], [1.0, 0.0, 1.0, 0.0, 1.0])
    assert len(xs) <= 5
    assert apply_isotonic(0.3, xs, ys) >= 0.0


def test_apply_isotonic_equal_knot_interval():
    assert apply_isotonic(0.5, (0.5, 0.5), (0.4, 0.6)) == 0.4


def test_apply_isotonic_interpolates_between_knots():
    val = apply_isotonic(0.35, (0.2, 0.8), (0.2, 0.8))
    assert val == 0.35


def test_apply_isotonic_single_knot_and_fallback():
    assert apply_isotonic(0.5, (), ()) == 0.5
    assert apply_isotonic(0.5, (0.5,), (0.7,)) == 0.7
    assert apply_isotonic(0.5, (0.5, 0.5), (0.4, 0.6)) == 0.4
    assert apply_isotonic(0.25, (0.3, 0.7), (0.2, 0.8)) == 0.2


def test_fit_calibrator_explicit_methods():
    probs = [0.9, 0.1, 0.8, 0.2]
    labels = [1.0, 0.0, 1.0, 0.0]
    assert fit_calibrator(probs, labels, calibration_cfg={"method": "temperature_platt"}).method == "temperature_platt"
    assert fit_calibrator(probs, labels, calibration_cfg={"method": "platt"}).method == "platt"
    assert fit_calibrator([], labels, calibration_cfg={"method": "platt"}).method == "identity"
    iso = fit_calibrator(
        probs,
        labels,
        calibration_cfg={"method": "isotonic", "isotonic_min_samples": 20},
    )
    assert iso.method in {"temperature_platt", "platt", "isotonic"}
    forced = fit_calibrator(
        probs,
        labels,
        calibration_cfg={"method": "auto", "auto_select_by_brier": False},
    )
    assert forced.method == "temperature_platt"
    sparse_iso = fit_calibrator(
        probs,
        labels,
        calibration_cfg={"method": "isotonic", "isotonic_min_samples": 20},
    )
    assert sparse_iso.method in {"temperature_platt", "platt", "isotonic"}


def test_select_best_calibrator_empty():
    assert _select_best_calibrator([]).method == "identity"


def test_median_tail_empty():
    assert _median_tail([], 8) == 0.0


def test_vol_component_sources():
    assert (
        _vol_component(
            vol_source="bb_width",
            bb_width=0.02,
            atr_norm=0.01,
            bb_baseline=0.01,
            atr_baseline=0.02,
        )
        >= 0.0
    )
    assert (
        _vol_component(
            vol_source="atr_norm",
            bb_width=0.02,
            atr_norm=0.01,
            bb_baseline=0.01,
            atr_baseline=0.02,
        )
        >= 0.0
    )


def test_volatility_regime_default_signal_path():
    score = volatility_regime_score(
        bb_width=0.018,
        atr_norm=0.017,
        adx=0.16,
        vol_ratio=0.98,
        bb_width_history=[0.018, 0.017, 0.019],
        atr_norm_history=[0.017, 0.016, 0.018],
        cfg={"baseline_lookback": 8},
    )
    assert 0.0 <= score <= 1.0


def test_fit_isotonic_monotonic_mapping():
    xs, ys = fit_isotonic([0.1, 0.3, 0.5, 0.7, 0.9], [0.0, 0.2, 0.5, 0.8, 1.0])
    assert len(xs) >= 2
    assert apply_isotonic(0.5, xs, ys) == 0.5


def test_apply_isotonic_edges():
    xs = (0.2, 0.8)
    ys = (0.1, 0.9)
    assert apply_isotonic(0.1, xs, ys) == 0.1
    assert apply_isotonic(0.9, xs, ys) == 0.9


def test_fit_calibrator_auto_selects_method():
    probs = [0.9, 0.8, 0.2, 0.1, 0.75, 0.25, 0.65, 0.35]
    labels = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    cal = fit_calibrator(probs, labels, calibration_cfg={"method": "auto", "isotonic_min_samples": 6})
    assert cal.method in {"temperature_platt", "platt", "isotonic", "identity"}
    calibrated = [apply_calibrator(p, cal) for p in probs]
    assert all(0.0 <= p <= 1.0 for p in calibrated)


def test_fit_calibrator_falls_back_to_identity_when_forced_collapses():
    from unittest.mock import patch

    from src.application.services.deep_learning.dl_sharpness import mean_sharpness

    probs = [0.62, 0.38, 0.71, 0.29, 0.68, 0.32, 0.77, 0.23, 0.66, 0.34] * 4
    labels = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] * 4
    collapsing = CalibratorState(method="temperature_platt", temperature=1.8, platt_a=0.05, platt_b=0.0)
    with patch(
        "src.application.services.deep_learning.dl_calibration_fit._build_temperature_platt",
        return_value=collapsing,
    ):
        cal = fit_calibrator(
            probs,
            labels,
            calibration_cfg={
                "method": "temperature_platt",
                "min_calibration_sharpness": 0.01,
            },
        )
    assert cal.method == "identity"
    assert mean_sharpness([apply_calibrator(p, cal) for p in probs]) >= 0.01 - 1e-9


def test_fit_calibrator_isotonic_explicit():
    probs = [0.1 + i * 0.1 for i in range(10)]
    labels = [1.0 if p >= 0.55 else 0.0 for p in probs]
    cal = fit_calibrator(probs, labels, calibration_cfg={"method": "isotonic", "isotonic_min_samples": 8})
    assert cal.method == "isotonic"
    assert len(cal.isotonic_x) >= 2


def test_calibrator_roundtrip_with_method():
    cal = CalibratorState(method="isotonic", isotonic_x=(0.2, 0.8), isotonic_y=(0.1, 0.9))
    payload = calibrator_to_dict(cal)
    restored = calibrator_from_dict(payload)
    assert restored.method == "isotonic"
    assert restored.isotonic_x == (0.2, 0.8)


def test_volatility_regime_directional_clean_low_score():
    score = volatility_regime_score(
        bb_width=0.02,
        atr_norm=0.015,
        adx=0.30,
        vol_ratio=0.95,
        bb_width_history=[0.02, 0.021, 0.019],
        atr_norm_history=[0.015, 0.014, 0.016],
        cfg={"directional_adx_min": 0.22, "baseline_lookback": 8},
    )
    assert score <= 0.35


def test_volatility_regime_compressive_high_score():
    score = volatility_regime_score(
        bb_width=0.005,
        atr_norm=0.030,
        adx=0.15,
        vol_ratio=1.20,
        bb_width_history=[0.02, 0.021, 0.019, 0.018],
        atr_norm_history=[0.010, 0.011, 0.012, 0.013],
        cfg={"compressive_bb_percentile": 0.25, "baseline_lookback": 8},
    )
    assert score >= 0.70


def test_resolve_dynamic_thresholds_high_regime_tightens():
    bundle = resolve_dynamic_thresholds(
        base_call=0.53,
        base_put=0.47,
        base_edge=0.04,
        regime_score=0.85,
        cfg={
            "high_regime_call_delta": 0.03,
            "high_regime_put_delta": 0.03,
            "high_regime_edge_delta": 0.015,
            "low_regime_call_delta": -0.02,
            "low_regime_put_delta": -0.02,
            "low_regime_edge_delta": -0.01,
        },
    )
    assert bundle.call_threshold > 0.53
    assert bundle.put_threshold < 0.47
    assert bundle.min_edge > 0.04


def test_resolve_dynamic_thresholds_low_regime_relaxes():
    bundle = resolve_dynamic_thresholds(
        base_call=0.53,
        base_put=0.47,
        base_edge=0.04,
        regime_score=0.15,
        cfg={
            "high_regime_call_delta": 0.03,
            "high_regime_put_delta": 0.03,
            "high_regime_edge_delta": 0.015,
            "low_regime_call_delta": -0.02,
            "low_regime_put_delta": -0.02,
            "low_regime_edge_delta": -0.01,
        },
    )
    assert bundle.call_threshold < 0.53
    assert bundle.put_threshold > 0.47
    assert bundle.min_edge < 0.04


def test_resolve_dynamic_threshold_bundle_disabled():
    assert (
        resolve_dynamic_threshold_bundle(
            base_call=0.53,
            base_put=0.47,
            base_edge=0.04,
            bb_width=0.02,
            atr_norm=0.01,
            adx=0.2,
            vol_ratio=1.0,
            cfg={"enabled": False},
        )
        is None
    )


def test_resolve_dynamic_threshold_bundle_vol_compression():
    bundle = resolve_dynamic_threshold_bundle(
        base_call=0.53,
        base_put=0.47,
        base_edge=0.04,
        bb_width=0.001,
        atr_norm=0.01,
        adx=0.19,
        vol_ratio=0.41,
        bb_width_history=[0.05, 0.04, 0.03],
        cfg={
            "enabled": True,
            "vol_compression_threshold": 0.50,
            "vol_compression_k_parabolic": 4.0,
            "vol_compression_k_hyperbolic": 0.15,
        },
        symbol="R_10",
    )
    assert bundle is not None
    assert bundle.min_edge > 0.04
