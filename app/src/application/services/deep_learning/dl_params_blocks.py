"""Blocos de parsing para parametros de deep learning."""

from typing import Any


def parse_calibration_config(dl_config: dict) -> dict[str, Any]:
    """Extrai configuracao do bloco deep_learning.calibration."""
    raw = dl_config.get("calibration") if isinstance(dl_config.get("calibration"), dict) else {}
    drift = raw.get("calibration_neutral_drift")
    if isinstance(drift, (list, tuple)) and len(drift) >= 2:
        neutral_drift = [float(drift[0]), float(drift[1])]
    else:
        half = float(raw.get("neutral_half_width", 0.02))
        neutral_drift = [0.5 - half, 0.5 + half]
    return {
        "method": str(raw.get("method", "auto")).strip().lower(),
        "isotonic_min_samples": max(3, int(raw.get("isotonic_min_samples", 20))),
        "auto_select_by_brier": bool(raw.get("auto_select_by_brier", True)),
        "entropy_ceiling": float(raw.get("entropy_ceiling", 0.92)),
        "entropy_penalty_strength": float(raw.get("entropy_penalty_strength", 1.0)),
        "entropy_floor": float(raw.get("entropy_floor", 0.0)),
        "calibration_neutral_drift": neutral_drift,
        "neutral_half_width": float(raw.get("neutral_half_width", 0.02)),
    }


def parse_indicator_gating_config(dl_config: dict) -> dict[str, Any]:
    """Extrai configuracao do bloco indicator_gating."""
    raw = dl_config.get("indicator_gating", {}) if isinstance(dl_config.get("indicator_gating"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "hurst_min": float(raw.get("hurst_min", 0.0)),
        "hurst_max": float(raw.get("hurst_max", 1.0)),
        "adx_min": float(raw.get("adx_min", 0.0)),
        "vol_ratio_min": float(raw.get("vol_ratio_min", 0.0)),
        "vol_ratio_max": float(raw.get("vol_ratio_max", 999.0)),
        "cmo_min": float(raw.get("cmo_min", -1.0)),
        "cmo_max": float(raw.get("cmo_max", 1.0)),
        "keltner_pct_b_min": float(raw.get("keltner_pct_b_min", -999.0)),
        "keltner_pct_b_max": float(raw.get("keltner_pct_b_max", 999.0)),
        "veto_on_noise": bool(raw.get("veto_on_noise", False)),
        "noise_hurst_lo": float(raw.get("noise_hurst_lo", 0.45)),
        "noise_hurst_hi": float(raw.get("noise_hurst_hi", 0.55)),
        "strong_trend_min": float(raw.get("strong_trend_min", 0.65)),
        "strong_revert_max": float(raw.get("strong_revert_max", 0.35)),
        "veto_missing_hurst": bool(raw.get("veto_missing_hurst", False)),
    }
