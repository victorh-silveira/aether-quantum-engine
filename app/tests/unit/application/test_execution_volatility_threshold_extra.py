from src.application.services.deep_learning.dl_calibration_fit import calibrator_entropy_metrics, fit_calibrator
from src.application.services.execution_volatility_threshold import resolve_dynamic_threshold_bundle


def test_resolve_dynamic_threshold_bundle_enabled():
    bundle = resolve_dynamic_threshold_bundle(
        base_call=0.53,
        base_put=0.47,
        base_edge=0.04,
        bb_width=0.02,
        atr_norm=0.01,
        adx=0.30,
        vol_ratio=0.95,
        bb_width_history=[0.02, 0.021],
        atr_norm_history=[0.01, 0.011],
        cfg={"enabled": True, "vol_source": "blend", "baseline_lookback": 8},
    )
    assert bundle is not None
    assert 0.51 <= bundle.call_threshold <= 0.62


def test_calibrator_entropy_metrics():
    probs = [0.9, 0.1, 0.8, 0.2]
    labels = [1.0, 0.0, 1.0, 0.0]
    cal = fit_calibrator(probs, labels, calibration_cfg={"method": "platt"})
    meta = calibrator_entropy_metrics(probs, labels, cal, calibration_cfg={"entropy_ceiling": 0.01})
    assert "calibrated_entropy" in meta
    assert isinstance(meta["entropy_violation"], bool)
