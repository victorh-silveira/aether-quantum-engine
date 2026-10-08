from src.application.services.execution_volatility_threshold import (
    resolve_dynamic_threshold_bundle,
    resolve_dynamic_thresholds,
    volatility_regime_score,
)


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
