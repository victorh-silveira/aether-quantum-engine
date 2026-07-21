"""Configuracao parametrica de dl params blocks."""

from typing import Any

from src.domain.config_knobs import (
    merge_settings_block,
    require_bool,
    require_float,
    require_int,
    require_keys,
)


_CALIBRATION_KEYS = (
    "method",
    "isotonic_min_samples",
    "auto_select_by_brier",
    "entropy_ceiling",
    "entropy_penalty_strength",
    "entropy_floor",
    "neutral_half_width",
)
_INDICATOR_GATING_KEYS = (
    "enabled",
    "hurst_min",
    "hurst_max",
    "adx_min",
    "vol_ratio_min",
    "vol_ratio_max",
    "cmo_min",
    "cmo_max",
    "keltner_pct_b_min",
    "keltner_pct_b_max",
    "veto_on_noise",
    "noise_hurst_lo",
    "noise_hurst_hi",
    "strong_trend_min",
    "strong_revert_max",
    "veto_missing_hurst",
)
_DYNAMIC_THRESHOLD_KEYS = (
    "enabled",
    "vol_source",
    "call_base",
    "put_base",
    "min_edge_base",
    "high_regime_call_delta",
    "high_regime_put_delta",
    "high_regime_edge_delta",
    "low_regime_call_delta",
    "low_regime_put_delta",
    "low_regime_edge_delta",
    "compressive_bb_percentile",
    "directional_adx_min",
    "baseline_lookback",
    "squeeze_edge_slope",
    "squeeze_edge_exponential_k",
    "squeeze_min_margin",
    "vol_compression_threshold",
    "vol_compression_k_parabolic",
    "vol_compression_k_hyperbolic",
    "require_indicator_consensus",
    "implied_vol_bb_scale",
    "clamp_call_min",
    "clamp_call_max",
    "clamp_put_min",
    "clamp_put_max",
    "clamp_edge_min",
    "clamp_edge_max",
    "bb_p10",
    "squeeze_vol_ratio_max",
    "hyperbolic_edge_floor",
    "hyperbolic_edge_cap",
)


def parse_calibration_config(dl_config: dict) -> dict[str, Any]:
    """Resolve calibration com merge de override parcial sobre o SSOT."""
    override = dl_config.get("calibration") if isinstance(dl_config, dict) else None
    raw = merge_settings_block(
        ("deep_learning", "calibration"),
        override if isinstance(override, dict) else None,
    )
    block = require_keys(raw, _CALIBRATION_KEYS, "deep_learning.calibration")
    drift = block.get("calibration_neutral_drift")
    if isinstance(drift, (list, tuple)) and len(drift) >= 2:
        neutral_drift = [float(drift[0]), float(drift[1])]
    else:
        half = require_float(block, "neutral_half_width")
        neutral_drift = [0.5 - half, 0.5 + half]
    return {
        "method": str(block["method"]).strip().lower(),
        "isotonic_min_samples": max(3, require_int(block, "isotonic_min_samples")),
        "auto_select_by_brier": require_bool(block, "auto_select_by_brier"),
        "entropy_ceiling": require_float(block, "entropy_ceiling"),
        "entropy_penalty_strength": require_float(block, "entropy_penalty_strength"),
        "entropy_floor": require_float(block, "entropy_floor"),
        "calibration_neutral_drift": neutral_drift,
        "neutral_half_width": require_float(block, "neutral_half_width"),
        "val_acc_trust_floor": require_float(block, "val_acc_trust_floor") if "val_acc_trust_floor" in block else None,
        "temperature_min": require_float(block, "temperature_min") if "temperature_min" in block else None,
        "temperature_max": require_float(block, "temperature_max") if "temperature_max" in block else None,
        "neutral_calibration_half_width": (
            require_float(block, "neutral_calibration_half_width")
            if "neutral_calibration_half_width" in block
            else None
        ),
        "tcn_macro_call_override": (
            require_float(block, "tcn_macro_call_override") if "tcn_macro_call_override" in block else None
        ),
        "tcn_macro_put_override": (
            require_float(block, "tcn_macro_put_override") if "tcn_macro_put_override" in block else None
        ),
    }


def parse_indicator_gating_config(dl_config: dict) -> dict[str, Any]:
    """Resolve indicator_gating com merge de override parcial sobre o SSOT."""
    override = dl_config.get("indicator_gating") if isinstance(dl_config, dict) else None
    raw = merge_settings_block(
        ("deep_learning", "indicator_gating"),
        override if isinstance(override, dict) else None,
    )
    block = require_keys(raw, _INDICATOR_GATING_KEYS, "deep_learning.indicator_gating")
    return {
        "enabled": require_bool(block, "enabled"),
        "hurst_min": require_float(block, "hurst_min"),
        "hurst_max": require_float(block, "hurst_max"),
        "adx_min": require_float(block, "adx_min"),
        "vol_ratio_min": require_float(block, "vol_ratio_min"),
        "vol_ratio_max": require_float(block, "vol_ratio_max"),
        "cmo_min": require_float(block, "cmo_min"),
        "cmo_max": require_float(block, "cmo_max"),
        "keltner_pct_b_min": require_float(block, "keltner_pct_b_min"),
        "keltner_pct_b_max": require_float(block, "keltner_pct_b_max"),
        "veto_on_noise": require_bool(block, "veto_on_noise"),
        "noise_hurst_lo": require_float(block, "noise_hurst_lo"),
        "noise_hurst_hi": require_float(block, "noise_hurst_hi"),
        "strong_trend_min": require_float(block, "strong_trend_min"),
        "strong_revert_max": require_float(block, "strong_revert_max"),
        "veto_missing_hurst": require_bool(block, "veto_missing_hurst"),
    }


def parse_dynamic_threshold_config(exec_config: dict) -> dict[str, Any]:
    """Resolve dynamic_threshold com merge de override parcial sobre o SSOT."""
    override = exec_config.get("dynamic_threshold") if isinstance(exec_config, dict) else None
    raw = merge_settings_block(
        ("orchestrator", "execution", "dynamic_threshold"),
        override if isinstance(override, dict) else None,
    )
    block = require_keys(raw, _DYNAMIC_THRESHOLD_KEYS, "orchestrator.execution.dynamic_threshold")
    return {
        "enabled": require_bool(block, "enabled"),
        "vol_source": str(block["vol_source"]).strip().lower(),
        "call_base": require_float(block, "call_base"),
        "put_base": require_float(block, "put_base"),
        "min_edge_base": require_float(block, "min_edge_base"),
        "high_regime_call_delta": require_float(block, "high_regime_call_delta"),
        "high_regime_put_delta": require_float(block, "high_regime_put_delta"),
        "high_regime_edge_delta": require_float(block, "high_regime_edge_delta"),
        "low_regime_call_delta": require_float(block, "low_regime_call_delta"),
        "low_regime_put_delta": require_float(block, "low_regime_put_delta"),
        "low_regime_edge_delta": require_float(block, "low_regime_edge_delta"),
        "compressive_bb_percentile": require_float(block, "compressive_bb_percentile"),
        "directional_adx_min": require_float(block, "directional_adx_min"),
        "baseline_lookback": max(8, require_int(block, "baseline_lookback")),
        "squeeze_edge_slope": require_float(block, "squeeze_edge_slope"),
        "squeeze_edge_exponential_k": require_float(block, "squeeze_edge_exponential_k"),
        "squeeze_min_margin": require_float(block, "squeeze_min_margin"),
        "vol_compression_threshold": require_float(block, "vol_compression_threshold"),
        "vol_compression_k_parabolic": require_float(block, "vol_compression_k_parabolic"),
        "vol_compression_k_hyperbolic": require_float(block, "vol_compression_k_hyperbolic"),
        "require_indicator_consensus": require_bool(block, "require_indicator_consensus"),
        "implied_vol_bb_scale": require_bool(block, "implied_vol_bb_scale"),
        "clamp_call_min": require_float(block, "clamp_call_min"),
        "clamp_call_max": require_float(block, "clamp_call_max"),
        "clamp_put_min": require_float(block, "clamp_put_min"),
        "clamp_put_max": require_float(block, "clamp_put_max"),
        "clamp_edge_min": require_float(block, "clamp_edge_min"),
        "clamp_edge_max": require_float(block, "clamp_edge_max"),
        "bb_p10": require_float(block, "bb_p10"),
        "squeeze_vol_ratio_max": require_float(block, "squeeze_vol_ratio_max"),
        "hyperbolic_edge_floor": require_float(block, "hyperbolic_edge_floor"),
        "hyperbolic_edge_cap": require_float(block, "hyperbolic_edge_cap"),
    }
