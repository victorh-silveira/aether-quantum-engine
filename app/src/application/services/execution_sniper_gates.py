"""Helpers de calibracao neutra para inferencia DL."""

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
    half = float(raw.get("neutral_half_width", 0.04))
    return 0.5 - half, 0.5 + half
