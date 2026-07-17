"""Gates sniper de calibracao neutra, Hurst e squeeze extremo."""

from __future__ import annotations

from typing import Any


def resolve_calibration_neutral_band(calibration_cfg: dict[str, Any] | None) -> tuple[float, float]:
    """Resolve banda neutra de calibracao a partir de drift ou half-width."""
    raw = calibration_cfg if isinstance(calibration_cfg, dict) else {}
    drift = raw.get("calibration_neutral_drift")
    if isinstance(drift, (list, tuple)) and len(drift) >= 2:
        lo = float(drift[0])
        hi = float(drift[1])
        if hi >= lo:
            return lo, hi
    half = float(raw.get("neutral_half_width", 0.02))
    return 0.5 - half, 0.5 + half


def hurst_regime_allowed(hurst: float | None, gating_cfg: dict[str, Any] | None) -> bool:
    """True quando o Hurst esta fora da zona de ruido ou dentro dos limites estaticos."""
    cfg = gating_cfg if isinstance(gating_cfg, dict) else {}
    if not bool(cfg.get("enabled", False)):
        return True
    if hurst is None:
        return not bool(cfg.get("veto_missing_hurst", False))
    value = float(hurst)
    if bool(cfg.get("veto_on_noise", False)):
        noise_lo = float(cfg.get("noise_hurst_lo", 0.45))
        noise_hi = float(cfg.get("noise_hurst_hi", 0.55))
        return not (noise_lo - 1e-12 <= value <= noise_hi + 1e-12)
    lo = float(cfg.get("hurst_min", 0.0))
    hi = float(cfg.get("hurst_max", 1.0))
    return lo - 1e-12 <= value <= hi + 1e-12


def apply_hurst_noise_veto(metrics: dict[str, Any], gating_cfg: dict[str, Any] | None) -> bool:
    """Veto de Hurst desativado; telemetria permanece disponivel via hurst_regime_allowed."""
    _ = (metrics, gating_cfg)
    return False


def apply_bb_squeeze_requirement(metrics: dict[str, Any], squeeze_cfg: dict[str, Any] | None) -> bool:
    """Exigencia de squeeze extremo desativada."""
    _ = (metrics, squeeze_cfg)
    return False
